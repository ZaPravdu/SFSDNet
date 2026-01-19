import os

from importlib import import_module

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms

from config import cfg
import datasets
from dataset_assembler import get_testset

from datasets.dataset import TestDataset
from model.VIC import Video_Counter

# load model


def calculate_mae(output: torch.Tensor, target: torch.Tensor):
    error = (target-output).abs()
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
    pseudo_dens_root = 'pseudo_density_map'
    error_root = 'SDNet_error_map'
    scene_path = './test.txt'
    device = 'cuda'
    scene_names = []

    with open(scene_path, 'r') as f:
        for line in f.readlines():
            scene_names.append(line.strip('\n'))

    state = torch.load('./sdnet.pth')
    new_state = {}
    for k, v in state.items():
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v

    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    model = Video_Counter(cfg, cfg_data)
    model.load_state_dict(new_state, strict=True)
    model.eval()
    model = model.to(device)
    # load data
    scene_path = './test.txt'
    dataset_path = './MovingDroneCrowd'

    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data
    test_loader = get_testset(dataset_path, scene_path, cfg_data, shuffle=False)
    # inference
    global_loss = []
    share_loss = []
    io_loss = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            images, targets = data
            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss = model(images.to(device),targets)

            gt_global_den = gt_global_den.detach() * cfg_data.DEN_FACTOR
            gt_share_den = gt_share_den.detach() * cfg_data.DEN_FACTOR
            gt_in_out_den = gt_in_out_den.detach() * cfg_data.DEN_FACTOR

            global_den_mae, global_den_mse = calculate_error(pre_global_den.cpu(), gt_global_den.detach().cpu())
            share_den_mae, share_den_mse = calculate_error(pre_share_den.cpu(), gt_share_den.detach().cpu())
            io_den_mae, io_den_mse = calculate_error(pre_in_out_den.cpu(), gt_in_out_den.detach().cpu())

            total_mae = np.stack([global_den_mae.numpy(), share_den_mae.numpy(), io_den_mae.numpy()])
            total_mse = np.stack([global_den_mse.numpy(), share_den_mse.numpy(), io_den_mse.numpy()])
            total_pseudo_dens = np.stack([pre_global_den.cpu().numpy(), pre_share_den.cpu().numpy(), pre_in_out_den.cpu().numpy()])

            assert targets[0]['scene_name'] == targets[1]['scene_name']

            scene, sub_scene = targets[0]['scene_name'].split('/')
            frame_pair = str(targets[0]['frame']) + str(targets[1]['frame'])
            file_name = f'{frame_pair}.npy'

            pseudo_dens_main_path = os.path.join(pseudo_dens_root, scene, sub_scene, 'density_map')
            mae_main_path = os.path.join(error_root, scene, sub_scene, 'mae')
            mse_main_path = os.path.join(error_root, scene, sub_scene, 'mse')

            for item in [pseudo_dens_main_path, mae_main_path, mse_main_path]:
                if not os.path.exists(item):
                    os.makedirs(item)
            global_loss.append(all_loss['global'])
            share_loss.append(all_loss['share'])
            io_loss.append(all_loss['in_out'])

            np.save(os.path.join(pseudo_dens_main_path, file_name), total_pseudo_dens)
            np.save(os.path.join(mae_main_path, file_name), total_mae)
            np.save(os.path.join(mse_main_path, file_name), total_mse)

    global_loss = np.stack(global_loss)
    share_loss = np.stack(share_loss)
    io_loss = np.stack(io_loss)
    total_loss = (global_loss+share_loss+io_loss)/3
    print('global:', global_loss.mean())
    print('share:', share_loss.mean())
    print('io:', io_loss.mean())
    print('total:', total_loss.mean())
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

