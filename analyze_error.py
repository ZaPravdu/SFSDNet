import json
import os
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import cfg

import datasets
import torch.nn.functional as F
from datasets.dataset import TestDataset
import torchvision.transforms as standard_transforms

from scipy.stats import pearsonr, spearmanr


def calculate_patch_error(error_map, patch_layout=(4, 4)):

    error_map = torch.Tensor(error_map)
    _, _, H, W = error_map.shape

    patch_errors = F.adaptive_avg_pool2d(error_map, output_size=patch_layout).numpy()

    return patch_errors


def calculate_correlation(a, b, mode='pearson'):

    a_flat = a.flatten()
    b_flat = b.flatten()

    if mode == 'pearson':
        correlation, p_value = pearsonr(a_flat, b_flat)
    if mode == 'spearman':
        correlation, p_value = spearmanr(a_flat, b_flat)

    return [float(correlation), float(p_value)]


def main():
    scene_path = './test.txt'
    dataset_path = './MovingDroneCrowd'
    dens_error_type = 'mae'
    recon_error_type = 'mae'
    correlation_mode = 'spearman'
    patch_layout = (8, 8)

    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data
    test_loader = get_testset(dataset_path, scene_path, cfg_data, shuffle=False)
    correlation_dict = {}
    for i, data in enumerate(tqdm(test_loader), 0):
        _, targets = data

        assert targets[0]['scene_name'] == targets[1]['scene_name']

        frame_pair = str(targets[0]['frame']) + str(targets[1]['frame'])
        file_name = f'{frame_pair}.npy'

        scene_name = targets[0]['scene_name']

        SDNet_error_path = os.path.join('SDNet_error_map', *scene_name.split('/'))
        VGGAE_error_path = os.path.join('VGGAE_error_map', *scene_name.split('/'))

        SDNet_error_path = os.path.join(SDNet_error_path, dens_error_type)
        VGGAE_error_path = os.path.join(VGGAE_error_path, recon_error_type)

        dens_error = np.load(os.path.join(SDNet_error_path, file_name))
        recon_error = np.load(os.path.join(VGGAE_error_path, file_name))

        global_dens_error, share_dens_error, io_dens_error = dens_error

        if patch_layout is not None:
            recon_error = calculate_patch_error(recon_error, patch_layout=patch_layout).mean(axis=1)[:, None, :, :]
            global_dens_error = calculate_patch_error(global_dens_error, patch_layout=patch_layout)
            share_dens_error = calculate_patch_error(share_dens_error, patch_layout=patch_layout)
            io_dens_error = calculate_patch_error(io_dens_error, patch_layout=patch_layout)

        p_global = calculate_correlation(recon_error, global_dens_error, mode=correlation_mode)
        p_io = calculate_correlation(recon_error, io_dens_error, mode=correlation_mode)
        p_share = calculate_correlation(recon_error, share_dens_error, mode=correlation_mode)

        if scene_name not in correlation_dict.keys():
            correlation_dict[scene_name] = {'p_global': [], 'p_io': [], 'p_share': []}

        correlation_dict[scene_name]['p_global'].append(p_global)
        correlation_dict[scene_name]['p_io'].append(p_io)
        correlation_dict[scene_name]['p_share'].append(p_share)

    with open(f'{correlation_mode}_dens-{dens_error_type}-vs-recon-{recon_error_type}_correlation_patch{patch_layout[0]}{patch_layout[1]}.json', 'w') as f:
        json.dump(correlation_dict, f)


if __name__ == '__main__':
    main()