import os

import numpy as np
from importlib import import_module

from torch.optim.lr_scheduler import CosineAnnealingLR

from config import cfg
from model.VIC import Video_Counter
import torch.optim
from pytorch_lightning import LightningModule

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class HyperModel(LightningModule):
    def __init__(self):
        super(HyperModel, self).__init__()

    def training_step(self, batch, batch_idx):
        data = batch
        loss = self.calculate_loss(data, type='train')
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        data = batch
        loss = self.calculate_loss(data, type='val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return {'val_loss': loss}

    def calculate_loss(self, data, type):
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
        self.freeze_backbone = freeze_backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        data_mode = cfg.DATASET
        datasetting = import_module(f'datasets.setting.{data_mode}')
        cfg_data = datasetting.cfg_data

        state = torch.load(weight_path)
        new_state = {}
        for k, v in state.items():
            name = k[7:] if k.startswith('module.') else k
            new_state[name] = v
        model = Video_Counter(cfg, cfg_data)
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
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.decode_layer2 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.decode_layer1 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 128),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            ConvBlock(64, 64),
            nn.Conv2d(64,3, kernel_size=3, padding='same'),
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

    def calculate_loss(self, data, type):
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
                 freeze_backbone=True, max_epochs=10):
        super().__init__()
        # self.orthogonal_loss = orthogonal_loss
        self.freeze_backbone = freeze_backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        data_mode = cfg.DATASET
        datasetting = import_module(f'datasets.setting.{data_mode}')
        cfg_data = datasetting.cfg_data

        state = torch.load(weight_path)
        new_state = {}
        for k, v in state.items():
            name = k[7:] if k.startswith('module.') else k
            new_state[name] = v

        self.model = Video_Counter(cfg, cfg_data)
        self.model.load_state_dict(new_state,
                                    strict=True)

        # if self.freeze_backbone:

        for p in self.model.global_decoder.parameters():
            p.requires_grad = False
        for p in self.model.share_decoder.parameters():
            p.requires_grad = False
        for p in self.model.in_out_decoder.parameters():
            p.requires_grad = False
        # for p in self.feature_fuse.parameters():
        #     p.requires_grad = False

        self.init_loss_mask()
    def forward(self, img, target):
        output = self.model(img, target)
        return output

    def init_loss_mask(self):
        pass
        # self.loss_masks={'scene':{'1': {'global_mask': None, 'share_mask'}}}
    def create_patch_mask(self, recon_error_map, patch_size=16, threshold_ratio=0.3, method='percentile'):
        """
        基于重建误差创建patch级别的mask

        Args:
            recon_error_map: 重建误差矩阵 [H, W]
            patch_size: patch大小
            threshold_ratio: 阈值比例 (0-1)
            method: 阈值选择方法 ('percentile', 'mean', 'median')

        Returns:
            patch_mask: 二值mask [H//patch_size, W//patch_size]
            pixel_mask: 上采样到原图大小的mask [H, W]
        """
        H, W = recon_error_map.shape

        # 确保尺寸能被patch_size整除
        H_patches = H // patch_size
        W_patches = W // patch_size
        H_crop = H_patches * patch_size
        W_crop = W_patches * patch_size

        # 裁剪到可整除的尺寸
        recon_error_crop = recon_error_map[:H_crop, :W_crop]

        # 重塑为patch视图 [n_patches_h, n_patches_w, patch_size, patch_size]
        patches = recon_error_crop.reshape(H_patches, patch_size, W_patches, patch_size)
        patches = patches.transpose(0, 2, 1, 3)  # [H_patches, W_patches, patch_size, patch_size]

        # 计算每个patch的平均重建误差 [H_patches, W_patches]
        patch_errors = np.mean(patches, axis=(2, 3))

        # 根据阈值方法确定阈值
        if method == 'percentile':
            threshold = np.percentile(patch_errors, threshold_ratio * 100)
        elif method == 'mean':
            threshold = np.mean(patch_errors) * threshold_ratio
        elif method == 'median':
            threshold = np.median(patch_errors) * threshold_ratio
        else:
            raise ValueError(f"未知的阈值方法: {method}")

        # 创建patch级别的mask (True表示可靠patch)
        patch_mask = patch_errors <= threshold

        # 将patch mask上采样到像素级别
        pixel_mask = np.kron(patch_mask, np.ones((patch_size, patch_size)))

        # 如果原图有不能整除的边界，填充为False
        if H_crop < H or W_crop < W:
            full_mask = np.zeros((H, W), dtype=bool)
            full_mask[:H_crop, :W_crop] = pixel_mask
            pixel_mask = full_mask

        # return patch_mask, pixel_mask, patch_errors
        return patch_mask

    def calculate_loss(self, data, type):
        img, target = data
        # assert data.shape == (2, 3, 768, 1024), f'data.shape: {data.shape}'
        pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss = self.forward(img, target)
        # loss_mask = self.create_patch_mask()

        # batch_idx = torch.arange(0, data.shape[0])
        # batch_idx = batch_idx.view(-1, 2)[:, [1, 0]].view_as(batch_idx)
        loss = 0
        for key in all_loss:
            loss += all_loss[key]
        # L_c = F.mse_loss(output, data[batch_idx])
        # self.log(type + '_recon_loss', L_c, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss.mean()


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
        x = F.relu(x, inplace=True)
        x = self.norm(x+shortcut)

        return x


if __name__ == '__main__':
    model = VGGAE()
    x = torch.randn(2, 3,512,512)
    output = model(x)
    print(output.size())