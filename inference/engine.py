import os

import numpy as np
import torch
from tqdm import tqdm

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


def mc_dropout_inference(data_loader, infer_cfg, model):
    model = model.train()
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


    global_loss = np.stack(global_loss)
    share_loss = np.stack(share_loss)
    io_loss = np.stack(io_loss)
    total_loss = (global_loss + share_loss + io_loss) / 3
    print('global:', global_loss.mean())
    print('share:', share_loss.mean())
    print('io:', io_loss.mean())
    print('total:', total_loss.mean())
def calculate_error(output, target):
    mae = (target-output).abs()
    mse = (target-output)**2
    return mae, mse

