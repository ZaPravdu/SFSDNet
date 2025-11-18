import os
from importlib import import_module

import numpy as np
import torch
from tqdm import tqdm
from config import cfg

from dataset_assembler import get_testset
from model_assembler import SFSDNet


def generate_error_map_path(scene, frame, error_type='mae', model_name='VGGAE'):
    scene, sub_scene = scene.split('/')

    main_path = os.path.join(f'{model_name}_error_map', scene, sub_scene, 'recon')
    file_name = f'{frame}_{error_type}.npy'
    path = os.path.join(main_path, file_name)
    return path


def create_patch_mask(recon_error_map, patch_size=16, threshold_ratio=0.3, method='percentile'):
    """
    基于重建误差创建patch级别的mask

    Args:
        recon_error_map: 重建误差矩阵 [H, W]
        patch_size: patch大小
        threshold_ratio: 阈值比例 (0-1)
        method: 阈值选择方法 ('percentile', 'mean', 'median')

    Returns:
        patch_mask: 二值mask [H//patch_size, W//patch_size]
        pixel_mask: 上采样到原图大小的mask [H, W]
    """
    recon_error_map = recon_error_map.squeeze()
    H, W = recon_error_map.shape

    # 确保尺寸能被patch_size整除
    H_patches = H // patch_size
    W_patches = W // patch_size
    H_crop = H_patches * patch_size
    W_crop = W_patches * patch_size

    # 裁剪到可整除的尺寸
    recon_error_crop = recon_error_map[:H_crop, :W_crop]

    # 重塑为patch视图 [n_patches_h, n_patches_w, patch_size, patch_size]
    patches = recon_error_crop.reshape(H_patches, patch_size, W_patches, patch_size)
    patches = patches.transpose(0, 2, 1, 3)  # [H_patches, W_patches, patch_size, patch_size]

    # 计算每个patch的平均重建误差 [H_patches, W_patches]
    patch_errors = np.mean(patches, axis=(2, 3))

    # 根据阈值方法确定阈值
    if method == 'percentile':
        threshold = np.percentile(patch_errors, threshold_ratio * 100)
    elif method == 'mean':
        threshold = np.mean(patch_errors) * threshold_ratio
    elif method == 'median':
        threshold = np.median(patch_errors) * threshold_ratio
    else:
        raise ValueError(f"未知的阈值方法: {method}")

    # 创建patch级别的mask (True表示可靠patch)
    patch_mask = patch_errors <= threshold

    # 将patch mask上采样到像素级别
    pixel_mask = np.kron(patch_mask, np.ones((patch_size, patch_size)))

    # 如果原图有不能整除的边界，填充为False
    if H_crop < H or W_crop < W:
        full_mask = np.zeros((H, W), dtype=bool)
        full_mask[:H_crop, :W_crop] = pixel_mask
        pixel_mask = full_mask

    # return patch_mask, pixel_mask, patch_errors
    return pixel_mask


def main():
    device = 'cuda'
    scene_path = './test.txt'
    dataset_path = './MovingDroneCrowd'

    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data
    # train_loader, sampler_train, val_loader, restore_transform = \
    #     datasets.loading_data(cfg.DATASET, 4, False, is_main_process())
    test_loader = get_testset(dataset_path, scene_path, cfg_data)

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
            for j, target in enumerate(targets, 0):
                # error_map_path = generate_error_map_path(target['scene_name'], target['frame'])
                # error_map = np.load(error_map_path)
                # loss_mask = create_patch_mask(error_map)
                pseudo_dens = np.concatenate([pre_global_den[j], pre_share_den[j], pre_in_out_den[j]])

                scene, sub_scene = target['scene_name'].split('/')
                frame = target['frame']
                file_name = f'{frame}.npy'
                pseudo_dens_main_path = os.path.join(pseudo_dens_root, scene, sub_scene)
                if not os.path.exists(pseudo_dens_main_path):
                    os.makedirs(pseudo_dens_main_path)

                pseudo_path = os.path.join(pseudo_dens_main_path, file_name)
                np.save(pseudo_path, pseudo_dens)


if __name__ == '__main__':
    main()