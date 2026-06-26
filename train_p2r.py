import sys
import datasets
from importlib import import_module
from config import cfg

from datasets.dataset import P2RDataset
from datasets.utils import get_testset
# from dataset_assembler import get_testset
from model.VIC import Video_Counter
import torch
import datasets
# from misc.tools import is_main_process

import os
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import model_assembler
import argparse
import random
import numpy as np

seed = 42

random.seed(seed)

# 2. 设置NumPy随机数生成器种子
np.random.seed(seed)

# 3. 设置PyTorch的CPU和GPU种子
torch.manual_seed(seed)  # 这个函数同时设置了CPU和所有GPU的种子


class TrainConfig():
    def __init__(self,):


        # self.group = self.experiment_name
        self.validate_mode = True
        self.resume = False
        if self.resume:
            self.ckpt_path = f'/home/mscs/houminqiu2/SFSDNet/weight/VIC/{self.experiment_name}/{self.experiment_name}-latest.ckpt'
        else:
            self.ckpt_path = None

        self.data_mode = 'MovingDroneCrowd'
        self.device = 'cuda'
        if self.data_mode == 'MovingDroneCrowd':
            self.dataset_path = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd'
            self.scene_path = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd/test.txt'
        elif self.data_mode.upper() == 'HT21':
            self.dataset_path = '/home/mscs/houminqiu2/SFSDNet/HT21'
            self.scene_path = '/home/mscs/houminqiu2/SFSDNet/HT21/test.txt'
        else:
            raise ValueError('Data mode error')
        # self.error_root = 'SDNet_error_map'
        self.model_path = '/home/mscs/houminqiu2/SFSDNet/sdnet.pth'

        ### Training
        self.lr = 0.0001
        self.weight_decay = 0
        self.shuffle=True
        self.max_epochs=5
        self.batch_size = 8

        ### Freeze
        self.freeze_backbone = True
        self.freeze_feature_fuse = True
        self.freeze_head = True
        self.freeze_attention = True
        self.weight_path ='/home/mscs/houminqiu2/SFSDNet/sdnet.pth'

        ### Method
        self.dens_recon = False
        self.ST = False
        self.partial=1
        self.reg_mode = 'l2'
        self.beta = 1
        self.use_attention_gate = True
        self.training_mode = 'p2r'
        self.delta_L_mode = 'exp'  # None=disabled, 'original', 'inv', 'exp'

        self.gt_ratios_per_scene = 0.3  # 0 表示不使用 GT 半监督模式；作用到每场景，向下取整至少1
        self.single_scene = None  # None=use all scenes, '25'=only scene 25
        self.pseudo = True
        # self.gate_freeze_json = 'low_importance_gates.json'


        ### Log name
        self.project_name = 'SFSDNet'
        # self.project_name='test'
        postfix = '-attn_gate' if self.use_attention_gate else ''
        self.experiment_name = f'{self.data_mode}-{self.reg_mode}-gt{self.gt_ratios_per_scene}-{self.single_scene if self.single_scene is not None else "all"}'+postfix
        if self.delta_L_mode is not None:
            self.experiment_name += f'-{self.delta_L_mode}_deltaL{self.beta}'
        if self.validate_mode:
            self.experiment_name = f'{self.data_mode}-SDNet-{self.single_scene if self.single_scene is not None else "all"}'
        # self.experiment_name = f'{self.data_mode}-source'
        

def get_callbacks(monitor, monitor_mode, project_name, experiment_name, patience=3):
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename='{epoch:02d}-{'+ monitor +':.4f}',
        save_top_k=1,
        mode=monitor_mode
    )
    # early_stopping = EarlyStopping(monitor=monitor, patience=patience, mode=monitor_mode)
    checkpoint_callback_latest = ModelCheckpoint(
        monitor=None,
        save_top_k=1,
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename=f'{experiment_name}-latest'
    )
    callbacks=[
        # early_stopping,
        checkpoint_callback,
        checkpoint_callback_latest
    ]
    return callbacks
# os.chdir('/home/mscs/houminqiu2/SFSDNet/')

def main():
    # os.environ['WANDB_MODE'] = 'offline'

    # config
    
    train_cfg = TrainConfig()
    datasetting = import_module(f'datasets.setting.{train_cfg.data_mode}')
    cfg_data = datasetting.cfg_data

    # DataLoader
    # scenes, restore_transform = datasets.loading_testset(data_mode, 1, False, mode='test')
    # train_loader = get_testset(train_cfg, P2RUnsupervisedDataset)
    train_loader, trainset = get_testset(train_cfg, P2RDataset, cfg_data, training=True)
    test_loader, _ = get_testset(train_cfg, P2RDataset, cfg_data)
    # trainset[0]

    # Logger
    fast_dev_run, wandb_logger = get_logger(train_cfg)
    model = model_assembler.P2RModel(cfg=cfg, cfg_data=cfg_data, train_loader=train_loader, **train_cfg.__dict__)


    # model = model_assembler.PseudoDensNet(train_cfg)
    # callbacks
    callbacks = get_callbacks(monitor='train_loss_epoch', monitor_mode='min', project_name=train_cfg.project_name,
                              experiment_name=train_cfg.experiment_name, patience=3)

    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=train_cfg.max_epochs, accelerator='gpu', gpus=1,
        logger=wandb_logger, default_root_dir=f'./weight/{train_cfg.project_name}',
        log_every_n_steps=1,
        precision=32,
        fast_dev_run=fast_dev_run,
        num_sanity_val_steps=0,
        accumulate_grad_batches=train_cfg.batch_size,
    )


    if train_cfg.validate_mode:
        trainer.validate(model, test_loader)
    else:
        trainer.fit(
            model, train_loader, test_loader,
            # ckpt_path= train_cfg.ckpt_path
        )


def get_logger(train_cfg):
    if train_cfg.project_name == 'test':
        # epochs = 1
        # train_loader.dataset = Subset(train_loader.dataset, range(20))
        # val_loader.dataset = Subset(val_loader.dataset, range(20))
        wandb_logger = None
        fast_dev_run = True
    else:
        # wandb.init(settings=wandb.Settings(_disable_stats=True), name=experiment_name, project=project_name, dir='./weight',
        #            mode='online')
        config_dict = vars(train_cfg)
        config_dict = {k: str(v) if isinstance(v, bool) else v for k, v in config_dict.items()}
        wandb_logger = WandbLogger(name=train_cfg.experiment_name, project=train_cfg.project_name, 
                                   save_dir='/home/mscs/houminqiu2/SFSDNet/weight', offline=False, settings=wandb.Settings(_disable_stats=True),
                                   config=config_dict)
        # wandb_logger = None
        fast_dev_run = False
    return fast_dev_run, wandb_logger


if __name__ == '__main__':
    main()

