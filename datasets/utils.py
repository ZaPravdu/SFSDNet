import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as standard_transforms

import datasets
from datasets.dataset import TestDataset


def get_testset(config):

    main_transform = datasets.train_resize_transform(config.cfg_data.TRAIN_SIZE[0], config.cfg_data.TRAIN_SIZE[1], flip=False)
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*config.cfg_data.MEAN_STD)
    ])
    scene_names = []
    with open(config.scene_path, 'r') as f:
        for line in f.readlines():
            scene_names.append(line.strip('\n'))
    last_scene_names = []
    for scene_name in scene_names:
        root = os.path.join(config.dataset_path, 'frames', scene_name)
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
                                  base_path=config.dataset_path,
                                  main_transform=main_transform,
                                  img_transform=img_transform,
                                  interval=config.cfg_data.VAL_FRAME_INTERVALS,
                                  skip_flag=False,
                                  target=True,
                                  datasetname='MovingDroneCrowd')
        fullset.append(sub_dataset)
    test_dataset = torch.utils.data.ConcatDataset(fullset)
    test_loader = DataLoader(test_dataset, shuffle=config.shuffle, batch_size=1, drop_last=False, num_workers=1,
                             collate_fn=datasets.collate_fn, persistent_workers=True)
    return test_loader
