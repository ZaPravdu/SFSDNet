import datasets
from importlib import import_module
from config import cfg
from datasets.utils import get_testset
# from dataset_assembler import get_testset
from model.VIC import Video_Counter
import torch
import datasets
from misc.tools import is_main_process

import os
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import model_assembler

import glob

class TrainConfig():
    def __init__(self):
        self.project_name = 'VIC'
        # project_name='test'
        self.experiment_name = 'VGGAE-T-no_flip-train'
        self.resume = False
        if self.resume:
            self.ckpt_path = f'weight/VIC/{self.experiment_name}/{self.experiment_name}-latest.ckpt'
        else:
            self.ckpt_path = None

        self.data_mode = cfg.DATASET
        self.datasetting = import_module(f'datasets.setting.{self.data_mode}')
        self.cfg_data = self.datasetting.cfg_data
        self.device = 'cuda'
        self.dataset_path = 'MovingDroneCrowd'
        self.pseudo_dens_root = 'pseudo_density_map'
        # self.error_root = 'SDNet_error_map'
        self.scene_path = './test.txt'
        self.model_path = './sdnet.pth'
        self.cfg = cfg
        self.shuffle = False
        self.epochs = 100
        self.batch_size = 64
        self.lr = 0.0001
        self.weight_decay = 1e-6
        self.shuffle=True
        self.freeze_backbone = True
        self.weight_path ='./sdnet.pth'
        # self.patch_layout = (8, 8)

torch.set_float32_matmul_precision('medium')

def get_callbacks(monitor, monitor_mode, project_name, experiment_name, patience=3):
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename='{epoch:02d}-{'+ monitor +':.4f}',
        save_top_k=1,
        mode=monitor_mode
    )
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, mode=monitor_mode)
    checkpoint_callback_latest = ModelCheckpoint(
        monitor=None,
        save_top_k=1,
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename=f'{experiment_name}-latest'
    )
    callbacks=[
        early_stopping,
        checkpoint_callback,
        checkpoint_callback_latest
    ]
    return callbacks


def main():
    # os.environ['WANDB_MODE'] = 'offline'

    # config
    train_cfg = TrainConfig()

    # model
    # model = model_assembler.VGGAE(lr=lr, weight_decay=weight_decay,
    #                                  freeze_backbone=freeze_backbone, max_epochs=epochs)
    model = model_assembler.SFSDNet(train_cfg, mask_type='mse')

    # DataLoader
    # scenes, restore_transform = datasets.loading_testset(data_mode, 1, False, mode='test')
    train_loader = get_testset(train_cfg)

    # Logger
    fast_dev_run, wandb_logger = get_logger(train_cfg)

    # callbacks
    callbacks = get_callbacks(monitor='train_loss_epoch', monitor_mode='min', project_name=train_cfg.project_name,
                              experiment_name=train_cfg.experiment_name, patience=3)

    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=train_cfg.epochs, accelerator='gpu',
        logger=wandb_logger, default_root_dir=f'./weight/{train_cfg.project_name}',
        log_every_n_steps=1,
        precision='bf16-mixed',
        fast_dev_run=fast_dev_run,
        num_sanity_val_steps=0
        # accumulate_grad_batches=4,
    )

    trainer.fit(
        model, train_loader, None,
        ckpt_path= train_cfg.ckpt_path
    )


def get_logger(train_cfg):
    if train_cfg.project_name == 'test':
        epochs = 1
        # train_loader.dataset = Subset(train_loader.dataset, range(20))
        # val_loader.dataset = Subset(val_loader.dataset, range(20))
        wandb_logger = None
        fast_dev_run = True
    else:
        # wandb.init(settings=wandb.Settings(_disable_stats=True), name=experiment_name, project=project_name, dir='./weight',
        #            mode='online')
        wandb_logger = WandbLogger(name=train_cfg.experiment_name, project=train_cfg.project_name, save_dir='./weight',
                                   id='1')
        # wandb_logger = None
        fast_dev_run = False
    return fast_dev_run, wandb_logger


if __name__ == '__main__':
    main()

