import datasets
from importlib import import_module
from config import cfg
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
    os.environ['WANDB_MODE'] = 'offline'

    # project_name = 'VIC'
    project_name='test'

    freeze_backbone = True

    lr = 0.0001
    weight_decay = 1e-6

    experiment_name = 'VGGAE-T'
    epochs = 20
    batch_size = 64
    # weight_path = ''
    model = model_assembler.VGGAE(lr=lr, weight_decay=weight_decay,
                                     freeze_backbone=freeze_backbone, max_epochs=epochs)

    # DataLoader
    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    val_frame_intervals = cfg_data.VAL_FRAME_INTERVALS
    distributed = False
    # scenes, restore_transform = datasets.loading_testset(data_mode, 1, False, mode='test')
    train_loader, sampler_train, val_loader, restore_transform = \
        datasets.loading_data(cfg.DATASET, val_frame_intervals, distributed, is_main_process())

    val_dataset = []

    for i, L in enumerate(val_loader):
        val_dataset.append(L[1].dataset)

    val_dataset = torch.utils.data.ConcatDataset(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False,
                                             collate_fn=datasets.collate_fn)

    if project_name == 'test':
        epochs = 1
        # train_loader.dataset = Subset(train_loader.dataset, range(20))
        # val_loader.dataset = Subset(val_loader.dataset, range(20))
        wandb_logger = None
        fast_dev_run = True
    else:
        wandb.init(settings=wandb.Settings(_disable_stats=True), name=experiment_name, project=project_name, dir='./weight',
                   mode='online')
        wandb_logger = WandbLogger(name=experiment_name, project=project_name, save_dir='./weight', )
        # wandb_logger = None
        fast_dev_run = False

    callbacks = get_callbacks(monitor='val_loss', monitor_mode='min', project_name=project_name,
                              experiment_name=experiment_name)

    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=epochs, accelerator='gpu',
        logger=wandb_logger, default_root_dir=f'./weight/{project_name}',
        log_every_n_steps=1,
        precision='bf16-mixed',
        fast_dev_run=fast_dev_run,
        # accumulate_grad_batches=2,
    )

    trainer.fit(
        model, train_loader, val_loader,
        # ckpt_path='weight/VIC/VGGAE-T/VGGAE-T-latest.ckpt'
    )


if __name__ == '__main__':
    main()

