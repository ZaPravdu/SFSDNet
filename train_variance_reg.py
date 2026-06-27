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
import argparse
import random
import numpy as np


_BOOL_FIELDS = [
    'shuffle', 'validate_mode', 'dens_recon', 'ST',
    'freeze_backbone', 'freeze_feature_fuse', 'freeze_head', 'freeze_attention',
    'use_attention_gate', 'pseudo', 'use_variance_reg',
]


def _conv_bool(args):
    for f in _BOOL_FIELDS:
        v = getattr(args, f, None)
        if v is not None:
            setattr(args, f, bool(v))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── 数据 ──
    parser.add_argument('--data-mode', type=str, default='MovingDroneCrowd')
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--scene-path', type=str, default=None)
    parser.add_argument('--partial', type=float, default=1.0)
    parser.add_argument('--shuffle', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=42)

    # ── 训练 ──
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)

    # ── 冻结 ──
    parser.add_argument('--freeze-backbone', type=int, default=1, choices=[0, 1])
    parser.add_argument('--freeze-feature-fuse', type=int, default=1, choices=[0, 1])
    parser.add_argument('--freeze-head', type=int, default=1, choices=[0, 1])
    parser.add_argument('--freeze-attention', type=int, default=1, choices=[0, 1])

    # ── 方法 ──
    parser.add_argument('--dens-recon', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ST', type=int, default=0, choices=[0, 1])
    parser.add_argument('--reg-mode', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--use-attention-gate', type=int, default=1, choices=[0, 1])
    parser.add_argument('--training-mode', type=str, default='p2r')
    parser.add_argument('--delta-L-mode', nargs='?', type=str, const='', default=None)
    parser.add_argument('--gt-ratios-per-scene', type=float, default=0.1)
    parser.add_argument('--single-scene', nargs='?', type=str, const='', default='')
    parser.add_argument('--pseudo', type=int, default=1, choices=[0, 1])
    parser.add_argument('--gate-freeze-json', type=str, default=None)
    parser.add_argument('--use-variance-reg', type=int, default=1, choices=[0, 1])

    # ── Stage 1 (variance reg 专属) ──
    parser.add_argument('--source-scene-path', type=str, default=None)
    parser.add_argument('--scene-finetune-epochs', type=int, default=1)

    # ── 路径/模式 ──
    parser.add_argument('--validate-mode', type=int, default=0, choices=[0, 1])
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--project-name', type=str, default='test')

    args = parser.parse_args()
    _conv_bool(args)
    _resolve_default_paths(args)
    args.experiment_name = _compute_experiment_name(args)
    return args


def _resolve_default_paths(args):
    if args.dataset_path is not None:
        return
    base = '/home/mscs/houminqiu2/SFSDNet'
    if args.data_mode == 'MovingDroneCrowd':
        args.dataset_path = f'{base}/MovingDroneCrowd'
        args.scene_path = f'{base}/MovingDroneCrowd/test.txt'
        if args.source_scene_path is None:
            args.source_scene_path = f'{base}/MovingDroneCrowd/train.txt'
    elif args.data_mode.upper() == 'HT21':
        args.dataset_path = f'{base}/HT21'
        args.scene_path = f'{base}/HT21/train.txt'
        if args.source_scene_path is None:
            args.source_scene_path = f'{base}/HT21/train.txt'
    else:
        raise ValueError(f'Unknown data_mode: {args.data_mode}')
    if args.weight_path is None:
        args.weight_path = f'{base}/sdnet.pth'


def _compute_experiment_name(args):
    if args.validate_mode:
        return f'{args.data_mode}-SDNet'
    postfix = '-attn_gate' if args.use_attention_gate else ''
    return f'{args.data_mode}-var_reg{postfix}-ep{args.scene_finetune_epochs}'


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


def get_logger(args):
    if args.project_name == 'test':
        return True, None
    config_dict = {k: str(v) if isinstance(v, bool) else v
                   for k, v in vars(args).items()}
    wandb_logger = WandbLogger(
        name=args.experiment_name,
        project=args.project_name,
        save_dir='/home/mscs/houminqiu2/SFSDNet/weight',
        offline=False,
        settings=wandb.Settings(_disable_stats=True),
        config=config_dict,
    )
    return False, wandb_logger


def main():
    args = parse_args()

    # ── 随机种子 ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasetting = import_module(f'datasets.setting.{args.data_mode}')
    cfg_data = datasetting.cfg_data

    # ====================================================================
    # Stage 1: Gate uncertainty estimation (per-scene fine-tuning)
    # ====================================================================
    print('=' * 60)
    print('Stage 1: Per-scene gate fine-tuning on source domain')
    print(f'  Source scenes: {args.source_scene_path}')
    print('=' * 60)

    # Build raw model + identity gates
    raw_model = Video_Counter(cfg, cfg_data)
    sd = torch.load(args.weight_path, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    raw_model.load_state_dict(sd, strict=True)
    for p in raw_model.parameters():
        p.requires_grad = False
    for name in ['share_decoder', 'global_decoder', 'in_out_decoder', 'Extractor', 'feature_fuse']:
        add_gates_to_conv(getattr(raw_model, name))
    if args.use_attention_gate:
        add_gates_to_attention(raw_model)

    # Per-scene loaders from source domain
    from types import SimpleNamespace
    src_config = SimpleNamespace(
        scene_path=args.source_scene_path,
        dataset_path=args.dataset_path,
        data_mode=args.data_mode,
        shuffle=False,
    )
    per_scene_loaders = get_per_scene_loaders(
        src_config, P2RDataset, cfg_data, training=True)

    reg_coeff_dict = gate_variance.estimate_gate_uncertainty(
        raw_model, per_scene_loaders,
        lr=args.lr,
        den_factor=cfg_data.DEN_FACTOR,
        epochs=args.scene_finetune_epochs,
    )

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

    train_loader, trainset = get_testset(args, P2RDataset, cfg_data, training=True)
    test_loader, _ = get_testset(args, P2RDataset, cfg_data)

    fast_dev_run, wandb_logger = get_logger(args)

    model = model_assembler.P2RModel(
        cfg=cfg, cfg_data=cfg_data,
        train_cfg=args, train_loader=train_loader,
    )
    model._apply_external_reg_coeff(reg_coeff_dict)

    callbacks = get_callbacks(
        monitor='train_loss_epoch',
        monitor_mode='min',
        project_name=args.project_name,
        experiment_name=args.experiment_name,
    )

    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        accelerator='gpu', gpus=1,
        logger=wandb_logger,
        default_root_dir=f'./weight/{args.project_name}',
        log_every_n_steps=1, precision=32,
        fast_dev_run=fast_dev_run, num_sanity_val_steps=0,
        accumulate_grad_batches=args.batch_size,
    )

    if args.validate_mode:
        trainer.validate(model, test_loader)
    else:
        trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
