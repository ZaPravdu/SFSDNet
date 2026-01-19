import os.path

import numpy as np
from importlib import import_module

from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import cfg

from model.VIC import Video_Counter
import torch.optim
from pytorch_lightning import LightningModule

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperModel(LightningModule):
    def __init__(self):
        super(HyperModel, self).__init__()

    def training_step(self, batch, batch_idx):
        data = batch
        loss = self.calculate_loss(data, mode='train')
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        data = batch
        loss = self.calculate_loss(data, mode='val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return {'val_loss': loss}

    def calculate_loss(self, data, mode):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,  # 余弦周期长度
                eta_min=self.lr*0.1  # 最小学习率（可选）
            ),
            'interval': 'epoch',  # 按epoch更新
            'frequency': 1,  # 每个epoch更新一次
            'name': 'cosine_annealing'  # 调度器名称（可选）
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


class VGGAE(HyperModel):
    def __init__(self, lr=0.0001, weight_decay=1e-6,
                 weight_path='./sdnet.pth',
                 freeze_backbone=True, max_epochs=10):
        super().__init__()
        # self.orthogonal_loss = orthogonal_loss
        self.reset_early_stop = False
        self.freeze_backbone = freeze_backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        data_mode = cfg.DATASET
        datasetting = import_module(f'datasets.setting.{data_mode}')
        cfg_data = datasetting.cfg_data
        model = Video_Counter(cfg, cfg_data)

        if weight_path is not None:
            state = torch.load(weight_path)
            new_state = {}
            for k, v in state.items():
                name = k[7:] if k.startswith('module.') else k
                new_state[name] = v

            model.load_state_dict(new_state,
                                    strict=True)

        self.backbone = model.Extractor
        self.share_cross_attention = model.share_cross_attention
        self.share_cross_attention_norm = model.share_cross_attention_norm
        self.feature_fuse = model.feature_fuse

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.share_cross_attention.parameters():
                p.requires_grad = False
            for p in self.share_cross_attention_norm.parameters():
                p.requires_grad = False
            for p in self.feature_fuse.parameters():
                p.requires_grad = False

        self.decode_layer3 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.decode_layer2 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.decode_layer1 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            ConvBlock(64, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding='same'),
        )

        self.init_weights(self.decode_layer1)
        self.init_weights(self.decode_layer2)
        self.init_weights(self.decode_layer3)

    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)

    def on_train_start(self):
        """训练开始时访问并修改 callback 状态"""
        print("=== 在模型类中重置 EarlyStopping ===")

        # 访问 trainer 中的 callbacks
        for callback in self.trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                print(f"找到 EarlyStopping callback: {callback.monitor}")

                if self.reset_early_stop:
                    # 重置状态
                    callback.best_score = torch.tensor(torch.inf)
                    callback.wait_count = 0
                    callback.stopped_epoch = 0

                    # 重置内部状态（不同版本兼容）
                    if hasattr(callback, '_best_score'):
                        callback._best_score = None
                    if hasattr(callback, '_wait_count'):
                        callback._wait_count = 0

                    print(f"✓ 已重置 {callback.monitor} 的状态")
                    print(f"  最佳分数: {callback.best_score}")
                    print(f"  等待计数: {callback.wait_count}")
                else:
                    print(f"保持 {callback.monitor} 的当前状态")
                    print(f"  最佳分数: {callback.best_score}")
                    print(f"  等待计数: {callback.wait_count}")

    def forward(self, img):
        features = self.backbone(img)
        B, C, H, W = features[-1].shape
        self.feature_scale = H / img.shape[2]

        img_pair_num = img.size(0) // 2
        assert img.size(0) % 2 == 0
        share_features = None
        for l_num in range(len(self.share_cross_attention)):
            share_results = []
            if share_features is not None:
                feature_fused = self.feature_fuse(share_features, features[l_num])

            for pair_idx in range(img_pair_num):
                if share_features is not None:
                    q1 = feature_fused[pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q1 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                kv = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                for i in range(len(self.share_cross_attention[l_num])):
                    q1 = self.share_cross_attention[l_num][i](q1, kv)
                    # if i == 0:
                    #     q1 = self.cross_pos_block(q1, H, W)

                q1 = self.share_cross_attention_norm(q1)

                if share_features is not None:
                    q2 = feature_fused[pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q2 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                kv = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                for i in range(len(self.share_cross_attention[l_num])):
                    q2 = self.share_cross_attention[l_num][i](q2, kv)
                    # if i == 0:
                    #     q2 = self.cross_pos_block(q2, H, W)

                q2 = self.share_cross_attention_norm(q2)

                share_results.append(q1)
                share_results.append(q2)

            share_features = torch.cat(share_results, dim=0)
            share_features = share_features.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        x = self.decode_layer3(share_features)
        x = self.decode_layer2(x)
        x = self.decode_layer1(x)

        return x

    def encode(self, x):
        x = self.backbone(x)
        features = torch.unflatten(x, 1, (512, 7, 7))
        # short_cut = features
        features = self.bottle_neck(features)
        # features = self.latent_norm(features + short_cut)

        return features

    def calculate_loss(self, data, mode):
        data = data[0]
        # assert data.shape == (2, 3, 768, 1024), f'data.shape: {data.shape}'
        output = self.forward(data)
        assert output.size()==data.size(), f'output: {output.size()}, data: {data.size()}'

        batch_idx = torch.arange(0, data.shape[0])
        batch_idx = batch_idx.view(-1, 2)[:, [1, 0]].view_as(batch_idx)
        L_c = F.mse_loss(output, data[batch_idx])
        # self.log(type + '_recon_loss', L_c, on_epoch=True, prog_bar=True, sync_dist=True)

        return L_c


class SFSDNet(HyperModel):
    def __init__(self, lr=0.0001, weight_decay=1e-6,
                 weight_path='./sdnet.pth',
                 freeze_backbone=True, max_epochs=10, mask_type='mse'):
        super().__init__()
        # self.orthogonal_loss = orthogonal_loss
        self.freeze_backbone = freeze_backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.mask_type = mask_type

        data_mode = cfg.DATASET
        datasetting = import_module(f'datasets.setting.{data_mode}')
        cfg_data = datasetting.cfg_data

        self.model = Video_Counter(cfg, cfg_data)
        if weight_path is not None:
            state = torch.load(weight_path)
            new_state = {}
            for k, v in state.items():
                name = k[7:] if k.startswith('module.') else k
                new_state[name] = v
            self.model.load_state_dict(new_state,
                                       strict=True)




        # if self.freeze_backbone:

        for p in self.model.global_decoder.parameters():
            p.requires_grad = False
        for p in self.model.share_decoder.parameters():
            p.requires_grad = False
        for p in self.model.in_out_decoder.parameters():
            p.requires_grad = False
        # for p in self.model.Extractor.parameters():
        #     p.requires_grad = False

    def forward(self, img, target):
        output = self.model(img, target)
        return output

    def calculate_loss(self, data, mode):
        images, targets = data

        # assert data.shape == (2, 3, 768, 1024), f'data.shape: {data.shape}'
        pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, gt_loss = self.forward(images, targets)
        # loss_mask = self.create_patch_mask()

        for key in gt_loss:
            self.log(f'gt_{key}_loss', gt_loss[key], on_epoch=True, prog_bar=True, sync_dist=True)

        pseudo_dens_map = []

        assert targets[0]['scene_name'] == targets[1]['scene_name']

        scene, sub_scene = targets[0]['scene_name'].split('/')
        frame_pair = str(targets[0]['frame']) + str(targets[1]['frame'])
        file_name = f'{frame_pair}.npy'
        pseudo_dens_main_path = os.path.join('pseudo_density_map', scene, sub_scene, 'density_map')
        pseudo_dens_path = os.path.join(pseudo_dens_main_path, file_name)
        pseudo_dens_map = np.load(pseudo_dens_path)

        pseudo_global_dens, pseudo_share_dens, pseudo_in_out_dens = torch.Tensor(pseudo_dens_map, device=self.device)

        # pseudo_dens_map = torch.Tensor(np.stack(pseudo_dens_map)*200)
        # pseudo_dens_map = pseudo_dens_map.to(self.device)
        # pseudo_dens_map.requires_grad = False

        # general_pre = torch.cat([pre_global_den, pre_share_den, pre_in_out_den], dim=1)
        if self.mask_type is not None:
            mask_main_path = os.path.join('pseudo_density_map', scene, sub_scene, f'{self.mask_type}_mask')
            loss_mask = torch.tensor(np.load(mask_main_path, file_name), device=self.device)
        else:
            loss_mask = torch.ones_like(pre_share_den, device=self.device, dtype=torch.bool)
        L_g = F.mse_loss(pre_global_den[loss_mask], pseudo_global_dens[loss_mask])
        L_s = F.mse_loss(pre_share_den[loss_mask], pseudo_share_dens[loss_mask])
        L_io = F.mse_loss(pre_in_out_den[loss_mask], pseudo_in_out_dens[loss_mask])

        loss_dict = {
            'global': L_g,
            'share': L_s,
            'in_out': L_io,
        }

        loss = 0
        for key in loss_dict.keys():
            self.log(f'pseudo_{key}_loss', loss_dict[key], on_epoch=True, prog_bar=True, sync_dist=True)
            loss += loss_dict[key]

        # L_c = F.mse_loss(output, data[batch_idx])
        # self.log(type + '_recon_loss', L_c, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss/3


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels==out_channels:
            self.res = True
        else:
            self.res = False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.norm = nn.GroupNorm(8, out_channels)

    def forward(self, x):
        if self.res:
            shortcut = x
        else:
            shortcut = 0

        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x+shortcut, inplace=True)

        return x


if __name__ == '__main__':
    model = VGGAE()
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print(output.size())