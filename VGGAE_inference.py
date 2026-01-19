import os

from importlib import import_module

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms

import model_assembler
from config import cfg
import datasets
from dataset_assembler import get_testset

from datasets.dataset import TestDataset
from model.VIC import Video_Counter

# load model


def calculate_mae(output: torch.Tensor, target: torch.Tensor):
    error = np.abs(target-output)
    return error


def calculate_mse(output: torch.Tensor, target: torch.Tensor):
    error = (target-output)**2
    return error


def calculate_error(output, target):
    mae = calculate_mae(output, target)
    mse = calculate_mse(output, target)
    return mae, mse


def save_npy(data, model_folder, scene, output_type, frame, error_type):
    scene_name, scene_id = scene.split('/')
    main_path = os.path.join(model_folder, scene_name, scene_id, output_type)
    file_name = f'{frame}_{error_type}.npy'
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    save_path = os.path.join(main_path, file_name)
    np.save(save_path, data)


def main():
    dataset_path = 'MovingDroneCrowd'
    scene_path = './test.txt'
    device = 'cuda'
    error_root = 'VGGAE_error_map'

    # model = model_assembler.VGGAE.load_from_checkpoint('weight/VIC/VGGAE-T-feature_merge-no_flip-train/epoch=91-train_loss_epoch=0.2490.ckpt')
    model = model_assembler.VGGAE.load_from_checkpoint(
        'weight/VIC/VGGAE-T-no_flip-train/epoch=99-train_loss_epoch=0.2158.ckpt')
    # model.eval()
    model = model.to(device)
    # load data
    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data
    test_loader = get_testset(dataset_path, scene_path, cfg_data, shuffle=False)
    # inference
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            images, targets = data
            recon = model(images.to(device))
            batch_idx = torch.arange(0, images.shape[0])
            batch_idx = batch_idx.view(-1, 2)[:, [1, 0]].view_as(batch_idx)
            recon_mae, recon_mse = calculate_error(recon.detach().cpu().float().numpy(),
                                                   images[batch_idx].detach().cpu().float().numpy())

            assert targets[0]['scene_name'] == targets[1]['scene_name']

            scene, sub_scene = targets[0]['scene_name'].split('/')
            frame_pair = str(targets[0]['frame']) + str(targets[1]['frame'])
            file_name = f'{frame_pair}.npy'

            mae_error_main_path = os.path.join(error_root, scene, sub_scene, 'mae')
            mse_error_main_path = os.path.join(error_root, scene, sub_scene, 'mse')

            for item in [mae_error_main_path, mse_error_main_path]:
                if not os.path.exists(item):
                    os.makedirs(item)

            np.save(os.path.join(mae_error_main_path, file_name), recon_mae)
            np.save(os.path.join(mse_error_main_path, file_name), recon_mse)

            # for j, target in enumerate(targets, 0):
            #     scene = target['scene_name']
            #     frame = target['frame']
            #
            #     save_npy(recon_mae[j].mean(axis=0, keepdims=True), 'VGGAE_error_map', scene, 'recon', frame, 'mae')
            #     save_npy(recon_mse[j].mean(axis=0, keepdims=True), 'VGGAE_error_map', scene, 'recon', frame, 'mse')

    # ae_path = 'weight/VIC/VGGAE/epoch=03-val_loss=0.7279.ckpt'
    # model = model_assembler.VGGAE.load_from_checkpoint(checkpoint_path=ae_path).cuda()
    # model.eval()


if __name__ == '__main__':
    main()
# for scene in scenes:
#     test_loader = scene[1]
#     for i, data in enumerate(tqdm(test_loader)):
#         images, targets = data
#         images = images.cuda()
#         recon = model(images)
#
#         recon_mae = calculate_mae(recon, images)
#         recon_mse = calculate_mse(recon, images)
#
#         for i, target in enumerate(targets, 0):
#             sample_errors[scene[0]][target['frame']]['recon_mae'] = recon_mae[i].detach().cpu().numpy()
#             sample_errors[scene[0]][target['frame']]['recon_mse'] = recon_mse[i].detach().cpu().numpy()




# calculate metrics for density map

# calculate metrics for reconstruction

# calculate correlation factor

