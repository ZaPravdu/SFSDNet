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

    model = model_assembler.VGGAE.load_from_checkpoint('./weight/VGGAE-T/epoch=03-val_loss=0.8124.ckpt')

    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')

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
            recon = model(images.to(device))
            recon_mae, recon_mse = calculate_error(recon.detach().cpu().numpy(), images.detach().cpu().numpy())

            for j, target in enumerate(targets, 0):
                scene = target['scene_name']
                frame = target['frame']

                save_npy(recon_mae[j], 'VGGAE_error_map', scene, 'recon', frame, 'mae')
                save_npy(recon_mse[j], 'VGGAE_error_map', scene, 'recon', frame, 'mse')


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

