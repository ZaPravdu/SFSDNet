"""
Training script: P2R semi-supervised learning with optional variance-based gate regularization.

Mode 1 — P2R training (default):
    Standard teacher-student P2R training with gate regularization.

Mode 2 — Variance-based gate regularization (--use-variance-reg 1):
    Stage 1: For each source-domain scene, fine-tune gates independently,
             collect gate values, compute cross-scene variance, derive reg_coeff.
    Stage 2: Inject the variance-based reg_coeff into a fresh P2RModel,
             then run normal P2R (or supervised) training.

Usage:
    python train_p2r.py
    python train_p2r.py --use-variance-reg 1 --scene-finetune-epochs 3
"""

import sys
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from importlib import import_module

import datasets
from config import cfg
from datasets.dataset import TTDADataset
from datasets.utils import get_per_scene_loaders, build_temporal_datasets
from model.VIC import Video_Counter
from model.gate_utils import add_gates_to_conv, add_gates_to_attention
import model.gate_variance as gate_variance
import model_assembler
import argparse


_BOOL_FIELDS = [
    'shuffle', 'validate_mode', 'dens_recon', 'ST',
    'freeze_backbone', 'freeze_feature_fuse', 'freeze_head', 'freeze_attention',
    'use_attention_gate', 'use_variance_reg', 'inject_gate',
]


def _conv_bool(args):
    for f in _BOOL_FIELDS:
        v = getattr(args, f, None)
        if v is not None:
            setattr(args, f, bool(v))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── 数据 ──
    parser.add_argument('--data-mode', type=str, default='MovingDroneCrowd', choices=['MovingDroneCrowd', 'HT21'])
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--scene-path', type=str, default=None)
    parser.add_argument('--partial', type=float, default=1.0)
    parser.add_argument('--shuffle', type=int, default=0, choices=[0, 1])
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
    parser.add_argument('--training-mode', type=str, default='semi_supervised',
                        choices=['supervised', 'semi_supervised', 'unsupervised'])
    parser.add_argument('--delta-L-mode', nargs='?', type=str, const=None, default=None, choices=['exp', 'original', 'inv'])
    parser.add_argument('--gt-ratios-per-scene', type=float, default=0.0)
    parser.add_argument('--single-scene', nargs='?', type=str, const='scene_25', default=None)
    parser.add_argument('--gate-freeze-json', type=str, default=None)
    parser.add_argument('--use-variance-reg', type=int, default=0, choices=[0, 1])
    parser.add_argument('--downsample-factor', type=int, default=1)
    parser.add_argument('--pseudo-mode', type=str, default='dens',
                        choices=['dens', 'feature', 'mixed'])
    parser.add_argument('--feature-pseudo-weight', type=float, default=1.0,
                        help='Weight for feature pseudo loss in mixed mode')
    parser.add_argument('--temporal-consist', type=int, default=1, choices=[0, 1],
                        help='Use frame buffer for temporal consistency')

    # ── Gate 控制 ──
    parser.add_argument('--inject-gate', type=int, default=1, choices=[0, 1])
    parser.add_argument('--gate-mode', type=str, default='independent',
                        choices=['independent', 'input_dependent', 'channel_mixture'])
    parser.add_argument('--prior-mean', type=int, default=1, choices=[0, 1])

    # ── 方差正则化（Stage 1） ──
    parser.add_argument('--source-scene-path', type=str, default=None)
    parser.add_argument('--scene-finetune-epochs', type=int, default=1)

    # ── 模型选择 ──
    parser.add_argument('--model', type=str, default='SDNet', choices=['SDNet', 'DRNet'])

    # ── 路径/模式 ──
    parser.add_argument('--validate-mode', type=int, default=0, choices=[0, 1])
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--project-name', type=str, default='SFSDNet')
    parser.add_argument('--experiment-suffix', type=str, default=None,
                        help='Suffix appended to experiment name (used by train_repeat.py)')

    args = parser.parse_args()
    _conv_bool(args)
    _resolve_default_paths(args)
    args.experiment_name = _compute_experiment_name(args)
    if args.experiment_suffix:
        args.experiment_name += args.experiment_suffix
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
        if getattr(args, 'model', 'SDNet') == 'DRNet':
            args.weight_path = f'{base}/SenseCrowd.pth'
        else:
            args.weight_path = f'{base}/sdnet.pth'


def _abbr_training_mode(mode):
    return {'supervised': 'sup', 'semi_supervised': 'semi', 'unsupervised': 'unsup'}.get(mode, mode)


def _format_beta(beta):
    if beta == int(beta):
        return str(int(beta))
    if 0 < beta < 0.01:
        s = f'{beta:.0e}'
        return s.replace('e-0', 'e-')   # 1e-03 → 1e-3
    return str(beta).rstrip('0').rstrip('.')


def _compute_experiment_name(args):
    if args.validate_mode:
        scene = args.single_scene if args.single_scene else 'all'
        return f'{args.data_mode}-SDNet-{scene}'
    if args.use_variance_reg:
        postfix = '-attn_gate' if args.use_attention_gate else ''
        return f'{args.data_mode}-var_reg{postfix}-ep{args.scene_finetune_epochs}'

    scene = f'{args.data_mode}-{args.single_scene}' if args.single_scene else args.data_mode
    train_mode = _abbr_training_mode(args.training_mode)
    beta_str = _format_beta(args.beta)
    w = args.feature_pseudo_weight
    if args.pseudo_mode != 'dens':
        pseudo_suffix = f'{w:.0f}' if w == int(w) else str(w)
    else:
        pseudo_suffix = ''
    pseudo_str = args.pseudo_mode + pseudo_suffix
    name = (
        f'{args.model}-gt{args.gt_ratios_per_scene}-{scene}'
        f'-gate-{train_mode}-reg{beta_str}'
        f'-{pseudo_str}-{args.prior_mean}{args.reg_mode}'
    )
    return name


def get_callbacks(monitor, monitor_mode, project_name, experiment_name, patience=3):
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename='{epoch:02d}-{'+ monitor +':.4f}',
        save_top_k=1,
        mode=monitor_mode,
    )
    # early_stopping = EarlyStopping(monitor=monitor, patience=patience, mode=monitor_mode)
    checkpoint_callback_latest = ModelCheckpoint(
        monitor=None,
        save_top_k=1,
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename=f'{experiment_name}-latest',
    )
    return [
        # early_stopping,
        checkpoint_callback,
        checkpoint_callback_latest,
    ]


def get_logger(args):
    if args.project_name == 'test':
        return True, None
    config_dict = {k: str(v) if isinstance(v, bool) else v for k, v in vars(args).items()}
    wandb_logger = WandbLogger(
        name=args.experiment_name, project=args.project_name,
        save_dir='/home/mscs/houminqiu2/SFSDNet/weight', offline=False,
        settings=wandb.Settings(_disable_stats=True), config=config_dict,
        group=os.environ.get('WANDB_RUN_GROUP'),
    )
    return False, wandb_logger


def main():
    args = parse_args()

    # ── 随机种子 ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # P2R 训练需要 SDNet 架构（默认为 GD3A，缺少 decoders / cross-attention 组件）
    cfg.MODEL = args.model

    # ── 数据集配置 ──
    datasetting = import_module(f'datasets.setting.{args.data_mode}')
    cfg_data = datasetting.cfg_data

    # ====================================================================
    # Stage 1: Gate uncertainty estimation (variance reg mode only)
    # ====================================================================
    if args.use_variance_reg:
        print('=' * 60)
        print('Stage 1: Per-scene gate fine-tuning on source domain')
        print(f'  Source scenes: {args.source_scene_path}')
        print('=' * 60)

        # Build a raw model for per-scene fine-tuning.
        if cfg.MODEL == 'DRNet':
            from model.drnet.vic import Video_Individual_Counter
            raw_model = Video_Individual_Counter(cfg, cfg_data).eval()
            sd = torch.load(args.weight_path, map_location='cpu')
            sd = {k.replace('Extractor.module.', 'Extractor.'): v
                  for k, v in sd.items()}
            raw_model.load_state_dict(sd, strict=True)
            for p in raw_model.parameters():
                p.requires_grad = False
            if args.inject_gate:
                add_gates_to_conv(raw_model.Extractor)
        else:
            raw_model = Video_Counter(cfg, cfg_data).eval()
            sd = torch.load(args.weight_path, map_location='cpu')
            sd = {k[7:] if k.startswith('module.') else k: v
                  for k, v in sd.items()}
            raw_model.load_state_dict(sd, strict=True)
            for p in raw_model.parameters():
                p.requires_grad = False
            if args.inject_gate:
                for name in ['share_decoder', 'global_decoder', 'in_out_decoder', 'Extractor', 'feature_fuse']:
                    add_gates_to_conv(getattr(raw_model, name))
                if args.use_attention_gate:
                    add_gates_to_attention(raw_model)

        from types import SimpleNamespace
        src_config = SimpleNamespace(
            scene_path=args.source_scene_path,
            dataset_path=args.dataset_path,
            data_mode=args.data_mode,
            shuffle=True,
        )
        per_scene_loaders = get_per_scene_loaders(
            src_config, TTDADataset, cfg_data, training=True)

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

        print('\n' + '=' * 60)
        print('Stage 2: P2R training with variance-based regularisation')
        print('=' * 60)

    # ====================================================================
    # Stage 2 / P2R 训练
    # ====================================================================
    train_dataset, val_dataset = build_temporal_datasets(args, cfg_data)
    train_loader = DataLoader(
        train_dataset, shuffle=False, batch_size=1, drop_last=False,
        num_workers=4, collate_fn=datasets.ttda_collate_fn, persistent_workers=True)
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, drop_last=False,
        num_workers=4, collate_fn=datasets.ttda_collate_fn, persistent_workers=True)

    fast_dev_run, wandb_logger = get_logger(args)
    model = model_assembler.get_model(
        cfg=cfg, cfg_data=cfg_data,
        train_cfg=args, train_loader=train_loader,
    )

    if args.use_variance_reg:
        model._apply_external_reg_coeff(reg_coeff_dict)

    callbacks = get_callbacks(
        monitor='train_loss_epoch', monitor_mode='min',
        project_name=args.project_name, experiment_name=args.experiment_name,
    )

    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=args.max_epochs, accelerator='gpu', gpus=1,
        logger=wandb_logger, default_root_dir=f'./weight/{args.project_name}',
        log_every_n_steps=1, precision=32,
        fast_dev_run=fast_dev_run, num_sanity_val_steps=0,
        accumulate_grad_batches=args.batch_size,
    )

    if args.validate_mode:
        trainer.validate(model, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)

    model.diagnose.save()


if __name__ == '__main__':
    main()
