import os
from importlib import import_module

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import cfg

from dataset_assembler import get_testset
from model_assembler import SFSDNet


def main():
    device = 'cuda'
    scene_path = './test.txt'
    dataset_path = './MovingDroneCrowd'

    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data
    test_loader = get_testset(dataset_path, scene_path, cfg_data)

    # train_loader, sampler_train, val_loader, restore_transform = \
    #     datasets.loading_data(cfg.DATASET, 4, False, is_main_process())
    model = SFSDNet()
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader), 0):
            images, targets = data
            # loss_masks = []
            pre_global_den, _, pre_share_den, _, pre_in_out_den, _, all_loss = model(images.to(device), targets)

            pre_global_den = pre_global_den.detach().cpu().numpy()
            pre_share_den = pre_share_den.detach().cpu().numpy()
            pre_in_out_den = pre_in_out_den.detach().cpu().numpy()

            pseudo_dens_root = 'pseudo_density_map'
            pseudo_dens = {
                'global': pre_global_den,
                'share': pre_share_den,
                'in_out': pre_in_out_den,
            }

            assert targets[0]['scene_name'] == targets[1]['scene_name']

            scene, sub_scene = targets[0]['scene_name'].split('/')
            frame_pair = str(targets[0]['frame']) + str(targets[1]['frame'])
            file_name = f'{frame_pair}.npy'
            pseudo_dens_main_path = os.path.join(pseudo_dens_root, scene, sub_scene, 'density_map')
            if not os.path.exists(pseudo_dens_main_path):
                os.makedirs(pseudo_dens_main_path)

            pseudo_path = os.path.join(pseudo_dens_main_path, file_name)
            np.save(pseudo_path, pseudo_dens)
            # for j, target in enumerate(targets, 0):
            #     # error_map_path = generate_error_map_path(target['scene_name'], target['frame'])
            #     # error_map = np.load(error_map_path)
            #     # loss_mask = create_patch_mask(error_map)
            #     pseudo_dens = np.concatenate([pre_global_den[j], pre_share_den[j], pre_in_out_den[j]])
            #
            #     scene, sub_scene = target['scene_name'].split('/')
            #     frame = target['frame']
            #     file_name = f'{frame}.npy'
            #     pseudo_dens_main_path = os.path.join(pseudo_dens_root, scene, sub_scene)
            #     if not os.path.exists(pseudo_dens_main_path):
            #         os.makedirs(pseudo_dens_main_path)
            #
            #     pseudo_path = os.path.join(pseudo_dens_main_path, file_name)
            #     np.save(pseudo_path, pseudo_dens)


if __name__ == '__main__':
    main()