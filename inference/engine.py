import json
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn.functional as F

from inference.analyzer import Analyzer


def pseudo_error_inference(data_loader, infer_cfg, model):
    global_loss = []
    share_loss = []
    io_loss = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            images, targets = data
            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss = model(
                images.to(infer_cfg.device), targets)

            gt_global_den = gt_global_den.detach() * infer_cfg.cfg_data.DEN_FACTOR
            gt_share_den = gt_share_den.detach() * infer_cfg.cfg_data.DEN_FACTOR
            gt_in_out_den = gt_in_out_den.detach() * infer_cfg.cfg_data.DEN_FACTOR

            global_den_mae, global_den_mse = calculate_error(pre_global_den.cpu(), gt_global_den.detach().cpu())
            share_den_mae, share_den_mse = calculate_error(pre_share_den.cpu(), gt_share_den.detach().cpu())
            io_den_mae, io_den_mse = calculate_error(pre_in_out_den.cpu(), gt_in_out_den.detach().cpu())

            total_mae = np.stack([global_den_mae.numpy(), share_den_mae.numpy(), io_den_mae.numpy()])
            total_mse = np.stack([global_den_mse.numpy(), share_den_mse.numpy(), io_den_mse.numpy()])
            total_pseudo_dens = np.stack(
                [pre_global_den.cpu().numpy(), pre_share_den.cpu().numpy(), pre_in_out_den.cpu().numpy()])

            assert targets[0]['scene_name'] == targets[1]['scene_name']

            scene, sub_scene = targets[0]['scene_name'].split('/')
            frame_pair = str(targets[0]['frame']) + str(targets[1]['frame'])
            file_name = f'{frame_pair}.npy'

            pseudo_dens_main_path = os.path.join(infer_cfg.pseudo_dens_root, scene, sub_scene, 'density_map')
            mae_main_path = os.path.join(infer_cfg.error_root, scene, sub_scene, 'mae')
            mse_main_path = os.path.join(infer_cfg.error_root, scene, sub_scene, 'mse')

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


def consistency_inference(data_loader, infer_cfg, model):
    model = model.eval()

    analyzer = Analyzer()
    all_mae_corr_dict = {}
    all_mse_corr_dict = {}
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            images, targets = data
            scene_name = targets[0]['scene_name']

            # Augmented inference
            transforms = [T.GaussianBlur(3, 0.1),
                          T.RandomHorizontalFlip(1),
                          T.ColorJitter(0.2, 0.2, 0.2)]
            forward_results = []
            # get forward groups
            for transform in transforms:
                augmented_images = transform(images)
                pre_global_den, _, pre_share_den, _, pre_in_out_den, _, _ = model(augmented_images.to(infer_cfg.device), targets)

                if isinstance(transform, T.RandomHorizontalFlip):
                    pre_global_den = transform(pre_global_den)
                    pre_share_den = transform(pre_share_den)
                    pre_in_out_den = transform(pre_in_out_den)

                forward_result = [pre_global_den, pre_share_den, pre_in_out_den]
                forward_results.append(forward_result)

            # original inference
            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, _ = model(images.to(infer_cfg.device), targets)
            forward_results.append([pre_global_den, pre_share_den, pre_in_out_den])
            forward_result = [[pre_global_den, pre_share_den, pre_in_out_den]]

            # Analyzer契约：接受所有的输出数据，并负责分拣和分析数据
            stds = analyzer.calculate_patch_uncertainty(forward_results)
            patch_mae_maps, patch_mse_maps = analyzer.analyze_patch_error(forward_result, [[gt_global_den, gt_share_den, gt_in_out_den]])
            mae_corr_dict = analyzer.calculate_correlation(stds, patch_mae_maps)
            mse_corr_dict = analyzer.calculate_correlation(stds, patch_mse_maps)

            if scene_name not in all_mae_corr_dict.keys():
                all_mae_corr_dict[scene_name] = {key: [] for key in mae_corr_dict.keys()}
                all_mse_corr_dict[scene_name] = {key: [] for key in mae_corr_dict.keys()}

            for channel_key in mae_corr_dict.keys():
                all_mae_corr_dict[scene_name][channel_key].append(mae_corr_dict[channel_key])
                all_mse_corr_dict[scene_name][channel_key].append(mse_corr_dict[channel_key])
            assert targets[0]['scene_name'] == targets[1]['scene_name']

    with open('mae.json', 'w') as f:
        json.dump(all_mae_corr_dict, f)
    with open('mse.json', 'w') as f:
        json.dump(all_mse_corr_dict, f)


            #
            # scene, sub_scene = targets[0]['scene_name'].split('/')
            # frame_pair = str(targets[0]['frame']) + str(targets[1]['frame'])
            # file_name = f'{frame_pair}.npy'
            #
            # pseudo_dens_main_path = os.path.join(infer_cfg.pseudo_dens_root, scene, sub_scene, 'density_map')
            # mae_main_path = os.path.join(infer_cfg.error_root, scene, sub_scene, 'mae')
            # mse_main_path = os.path.join(infer_cfg.error_root, scene, sub_scene, 'mse')
            #
            # for item in [pseudo_dens_main_path, mae_main_path, mse_main_path]:
            #     if not os.path.exists(item):
            #         os.makedirs(item)
            # global_loss.append(all_loss['global'])
            # share_loss.append(all_loss['share'])
            # io_loss.append(all_loss['in_out'])


    # global_loss = np.stack(global_loss)
    # share_loss = np.stack(share_loss)
    # io_loss = np.stack(io_loss)
    # total_loss = (global_loss + share_loss + io_loss) / 3
    # print('global:', global_loss.mean())
    # print('share:', share_loss.mean())
    # print('io:', io_loss.mean())
    # print('total:', total_loss.mean())


