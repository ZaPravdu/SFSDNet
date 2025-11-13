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
                 weight_path='./ep_120_iter_105000_mae_10.119_mse_13.722_seq_MAE_29.751_WRAE_25.237_MIAE_3.157_MOAE_2.663.pth',
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

        state = torch.load(weight_path, map_location=torch.device('cpu'))

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