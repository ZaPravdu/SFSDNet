"""
Two-stage training script: variance-based gate regularization.

Stage 1 — Gate uncertainty estimation:
    For each source-domain scene, fine-tune gates independently
    from identity. Collect gate values across scenes, compute
    cross-scene variance, derive reg_coeff = 1/variance.
    (Ridge regression: high variance = uncertain = small reg_coeff.)

Stage 2 — P2R training:
    Inject the variance-based reg_coeff into a fresh P2RModel,
    then run normal P2R (or supervised) training.

Usage:
    python train_variance_reg.py
"""

import os
import sys
from importlib import import_module

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

import datasets
from config import cfg
from datasets.dataset import P2RDataset
from datasets.utils import get_testset, get_per_scene_loaders
from model.VIC import Video_Counter
from model.gate_utils import add_gates_to_conv, add_gates_to_attention
import model.gate_variance as gate_variance
import model_assembler
import random
import numpy as np


seed = 42

random.seed(seed)

# 2. 设置NumPy随机数生成器种子
np.random.seed(seed)

# 3. 设置PyTorch的CPU和GPU种子
torch.manual_seed(seed)  # 这个函数同时设置了CPU和所有GPU的种子


class TrainConfig:
    """Configuration for variance-reg training (inherits train_p2r.py pattern)."""
    def __init__(self):
        self.validate_mode = False
        self.resume = False
        self.ckpt_path = None

        self.data_mode = 'MovingDroneCrowd'
        self.device = 'cuda'
        if self.data_mode == 'MovingDroneCrowd':
            self.dataset_path = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd'
            self.scene_path = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd/test.txt'
            self.source_scene_path = os.path.join(self.dataset_path, 'train.txt')
        elif self.data_mode.upper() == 'HT21':
            self.dataset_path = '/home/mscs/houminqiu2/SFSDNet/HT21'
            self.scene_path = '/home/mscs/houminqiu2/SFSDNet/HT21/train.txt'
            self.source_scene_path = os.path.join(self.dataset_path, 'train.txt')
        else:
            raise ValueError('Data mode error')

        self.weight_path = '/home/mscs/houminqiu2/SFSDNet/sdnet.pth'

        ### Training (Stage 2)
        self.lr = 0.0001
        self.weight_decay = 0
        self.shuffle = True
        self.max_epochs = 5
        self.batch_size = 8

        ### Freeze
        self.freeze_backbone = True
        self.freeze_feature_fuse = True
        self.freeze_head = True
        self.freeze_attention = True

        ### Method
        self.dens_recon = False
        self.ST = False
        self.partial = 1
        self.reg_mode = 'l2'
        self.beta = 1
        self.use_attention_gate = True
        self.training_mode = 'p2r'
        self.delta_L_mode = None    # variance reg uses external coeff, not delta-L
        self.use_variance_reg = True   # enables variance-based reg_coeff

        self.gt_ratios_per_scene = 0.1
        self.pseudo = True

        ### Stage 1 (per-scene fine-tuning)
        self.scene_finetune_epochs = 1

        ### Log name
        # self.project_name = 'SFSDNet'
        self.project_name = 'test'

        postfix = '-attn_gate' if self.use_attention_gate else ''
        self.experiment_name = f'{self.data_mode}-var_reg{postfix}-ep{self.scene_finetune_epochs}'
        if self.validate_mode:
            self.experiment_name = f'{self.data_mode}-SDNet'


def get_callbacks(monitor, monitor_mode, project_name, experiment_name):
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename='{epoch:02d}-{' + monitor + ':.4f}',
        save_top_k=1,
        mode=monitor_mode,
    )
    checkpoint_callback_latest = ModelCheckpoint(
        monitor=None,
        save_top_k=1,
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename=f'{experiment_name}-latest',
    )
    return [checkpoint_callback, checkpoint_callback_latest]


def get_logger(train_cfg):
    if train_cfg.project_name == 'test':
        return True, None
    config_dict = {k: str(v) if isinstance(v, bool) else v
                   for k, v in vars(train_cfg).items()}
    wandb_logger = WandbLogger(
        name=train_cfg.experiment_name,
        project=train_cfg.project_name,
        save_dir='/home/mscs/houminqiu2/SFSDNet/weight',
        offline=False,
        settings=wandb.Settings(_disable_stats=True),
        config=config_dict,
    )
    return False, wandb_logger


def main():
    train_cfg = TrainConfig()
    datasetting = import_module(f'datasets.setting.{train_cfg.data_mode}')
    cfg_data = datasetting.cfg_data

    # ====================================================================
    # Stage 1: Gate uncertainty estimation (per-scene fine-tuning)
    # ====================================================================
    print('=' * 60)
    print('Stage 1: Per-scene gate fine-tuning on source domain')
    print(f'  Source scenes: {train_cfg.source_scene_path}')
    print('=' * 60)

    # Build raw model + identity gates (only in submodules that Stage 2 will gate)
    raw_model = Video_Counter(cfg, cfg_data)
    sd = torch.load(train_cfg.weight_path, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    raw_model.load_state_dict(sd, strict=True)
    for p in raw_model.parameters():
        p.requires_grad = False
    # Match P2RModel._inject_gates() exactly — only these 5 submodules get conv gates
    for name in ['share_decoder', 'global_decoder', 'in_out_decoder', 'Extractor', 'feature_fuse']:
        add_gates_to_conv(getattr(raw_model, name))
    if train_cfg.use_attention_gate:
        add_gates_to_attention(raw_model)

    # Per-scene loaders from source domain (train.txt scenes)
    from types import SimpleNamespace
    src_config = SimpleNamespace(
        scene_path=train_cfg.source_scene_path,
        dataset_path=train_cfg.dataset_path,
        data_mode=train_cfg.data_mode,
        shuffle=False,
    )
    per_scene_loaders = get_per_scene_loaders(
        src_config, P2RDataset, cfg_data, training=True)

    reg_coeff_dict = gate_variance.estimate_gate_uncertainty(
        raw_model, per_scene_loaders,
        lr=train_cfg.lr,
        den_factor=cfg_data.DEN_FACTOR,
        epochs=train_cfg.scene_finetune_epochs,
    )

    # Clean up Stage 1 model and data loaders to free memory
    del raw_model, sd, per_scene_loaders, src_config
    torch.cuda.empty_cache()

    n_gates = len(reg_coeff_dict)
    n_params = sum(len(v) for v in reg_coeff_dict.values())
    print(f'\n[Stage 1] Estimated uncertainty for {n_gates} gate params '
          f'({n_params} total channels/heads)')

    # ====================================================================
    # Stage 2: P2R training with variance-based reg_coeff
    # ====================================================================
    print('\n' + '=' * 60)
    print('Stage 2: P2R training with variance-based regularisation')
    print('=' * 60)

    # Data loaders (target domain)
    train_loader, trainset = get_testset(
        train_cfg, P2RDataset, cfg_data, training=True)
    test_loader, _ = get_testset(train_cfg, P2RDataset, cfg_data)

    # Logger + callbacks
    fast_dev_run, wandb_logger = get_logger(train_cfg)

    # P2RModel with variance-based reg_coeff
    model = model_assembler.P2RModel(
        cfg=cfg, cfg_data=cfg_data,
        train_loader=train_loader,
        **train_cfg.__dict__,
    )
    model._apply_external_reg_coeff(reg_coeff_dict)

    callbacks = get_callbacks(
        monitor='train_loss_epoch',
        monitor_mode='min',
        project_name=train_cfg.project_name,
        experiment_name=train_cfg.experiment_name,
    )

    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=train_cfg.max_epochs,
        accelerator='gpu',
        gpus=1,
        logger=wandb_logger,
        default_root_dir=f'./weight/{train_cfg.project_name}',
        log_every_n_steps=1,
        precision=32,
        fast_dev_run=fast_dev_run,
        num_sanity_val_steps=0,
        accumulate_grad_batches=train_cfg.batch_size,
    )

    if train_cfg.validate_mode:
        trainer.validate(model, test_loader)
    else:
        trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
