import os

from importlib import import_module

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms

from config import cfg
import datasets

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
    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    main_transform = datasets.train_resize_transform(cfg_data.TRAIN_SIZE[0], cfg_data.TRAIN_SIZE[1], flip=False)
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    last_scene_names = []
    for scene_name in scene_names:
        root = os.path.join(dataset_path, 'frames', scene_name)
        if '/' in scene_name:
            scene_name, clip_names = scene_name.split('/')
            clip_names = [clip_names]
        else:
            clip_names = [clip_name for clip_name in os.listdir(root) if not '.' in clip_name]
            clip_names.sort()

        for clip_name in clip_names:
            scene_clip_name = "{}/{}".format(scene_name, clip_name)
            last_scene_names.append(scene_clip_name)
    scene_names = last_scene_names

    fullset = []
    for scene in scene_names:
        sub_dataset = TestDataset(scene_name=scene,
                                  base_path=dataset_path,
                                  main_transform=main_transform,
                                  img_transform=img_transform,
                                  interval=cfg_data.VAL_FRAME_INTERVALS,
                                  skip_flag=False,
                                  target=True,
                                  datasetname='MovingDroneCrowd')
        fullset.append(sub_dataset)

    test_dataset = torch.utils.data.ConcatDataset(fullset)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=False, num_workers=1,
                             collate_fn=datasets.collate_fn)

    # inference
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            images, targets = data
            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss = model(images.to(device),targets)

            global_den_mae, global_den_mse = calculate_error(pre_global_den.detach().cpu().numpy(), gt_global_den.detach().cpu().numpy())
            share_den_mae, share_den_mse = calculate_error(pre_share_den.detach().cpu().numpy(), gt_share_den.detach().cpu().numpy())
            io_den_mae, io_den_mse = calculate_error(pre_in_out_den.detach().cpu().numpy(), gt_in_out_den.detach().cpu().numpy())

            for j, target in enumerate(targets, 0):
                scene = target['scene_name']
                frame = target['frame']

                save_npy(global_den_mae[j], 'SDNet_error_map', scene, 'global', frame, 'mae')
                save_npy(global_den_mse[j], 'SDNet_error_map', scene, 'global', frame, 'mse')
                save_npy(share_den_mae[j], 'SDNet_error_map', scene, 'global', frame, 'mae')
                save_npy(share_den_mse[j], 'SDNet_error_map', scene, 'global', frame, 'mse')
                save_npy(io_den_mae[j], 'SDNet_error_map', scene, 'global', frame, 'mae')
                save_npy(io_den_mse[j], 'SDNet_error_map', scene, 'global', frame, 'mse')


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

