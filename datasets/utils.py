import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as standard_transforms

import datasets
import random
import math
from datasets.dataset import P2RDataset
# from datasets.dataset import P2RDataset, TestDataset


# if __name__ == '__main__':
    # from train_script import main
    

def get_testset(config, Dataset, cfg_data, training=False):
    main_transform = datasets.train_resize_transform(cfg_data.TRAIN_SIZE[0], cfg_data.TRAIN_SIZE[1], flip=False)
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])
    # resolve dataset name from config
    datasetname = config.data_mode if config.data_mode == 'MovingDroneCrowd' else config.data_mode.upper()
    scene_names = []
    with open(config.scene_path, 'r') as f:
        for line in f.readlines():
            scene_names.append(line.strip('\n'))

    # Single scene fine-tuning: only keep the specified scene
    if hasattr(config, 'single_scene') and config.single_scene:
        matched = [s for s in scene_names if s == config.single_scene]
        assert matched, (
            f"Single scene '{config.single_scene}' not found in {config.scene_path}. "
            f"Available scenes: {scene_names[:10]}{'...' if len(scene_names) > 10 else ''}"
        )
        scene_names = matched
    if datasetname == 'MovingDroneCrowd':
        # MDC: scenes have sub-clips under frames/ directory
        last_scene_names = []
        for scene_name in scene_names:
            root = os.path.join(config.dataset_path, 'frames', scene_name)
            if '/' in scene_name:
                scene_name, clip_names = scene_name.split('/')
                clip_names = [clip_names]
            else:
                clip_names = [clip_name for clip_name in os.listdir(root) if '.' not in clip_name]
                clip_names.sort()

            for clip_name in clip_names:
                scene_clip_name = "{}/{}".format(scene_name, clip_name)
                last_scene_names.append(scene_clip_name)
        scene_names = last_scene_names
    # for HT21 (and other datasets): scene names already contain full subpath
    # e.g. 'train/HT21-01', no frames/ dir, no sub-clip expansion
    fullset = []
    for scene in scene_names:
        kwargs = dict(gt_ratio=getattr(config, 'gt_ratios_per_scene', 0.0)) \
            if Dataset == datasets.dataset.P2RDataset else {}
        sub_dataset = Dataset(scene_name=scene,
                                  base_path=config.dataset_path,
                                  main_transform=main_transform,
                                  img_transform=img_transform,
                                  interval=cfg_data.VAL_FRAME_INTERVALS,
                                  skip_flag=False,
                                  target=True,
                                  datasetname=datasetname,
                                  training=training,
                                  **kwargs)
        fullset.append(sub_dataset)
    test_dataset = torch.utils.data.ConcatDataset(fullset)
    
    # 如果指定了partial参数，则从完整数据集中随机选择部分样本

    
    if Dataset == datasets.dataset.P2RDataset:
        if config.partial is not None and 0 < config.partial < 1:
            # 计算要选择的样本数量
            total_samples = len(test_dataset)
            selected_samples_count = math.ceil(total_samples * config.partial)
            
            # 随机选择索引
            indices = list(range(total_samples))
            random.shuffle(indices)
            selected_indices = indices[:selected_samples_count]
            
            # 创建Subset使用选中的索引
            test_dataset = torch.utils.data.Subset(test_dataset, selected_indices)
        test_loader = DataLoader(test_dataset, shuffle=config.shuffle, batch_size=1, drop_last=False, num_workers=4,
                             collate_fn=datasets.p2r_collate_fn, persistent_workers=True)
    elif Dataset== datasets.dataset.TestDataset:
        if config.partial is not None and 0 < config.partial < 1:
            # 计算要选择的样本数量
            total_samples = len(test_dataset)
            selected_samples_count = math.ceil(total_samples * config.partial)
            
            # 随机选择索引
            indices = list(range(total_samples))
            random.shuffle(indices)
            selected_indices = indices[:selected_samples_count]
            
            # 创建Subset使用选中的索引
            test_dataset = torch.utils.data.Subset(test_dataset, selected_indices)
        test_loader = DataLoader(test_dataset, shuffle=config.shuffle, batch_size=1, drop_last=False, num_workers=1,
                             collate_fn=datasets.collate_fn, persistent_workers=True)
    else:
        raise NotImplementedError("Should not happen...")
    return test_loader, test_dataset


def get_per_scene_loaders(config, Dataset, cfg_data, training=False):
    """
    Create one DataLoader per base scene for per-scene gate fine-tuning.

    Unlike get_testset() which concatenates all scenes into a single loader,
    this function groups sub-clips under the same base scene and returns
    a dict of {scene_name: DataLoader}.

    Args:
        config: object with .scene_path, .dataset_path, .data_mode, .shuffle
        Dataset: dataset class (P2RDataset or TestDataset)
        cfg_data: dataset config from datasets.setting
        training: passed through to Dataset constructor

    Returns:
        dict[base_scene_name, DataLoader]
    """
    main_transform = datasets.train_resize_transform(
        cfg_data.TRAIN_SIZE[0], cfg_data.TRAIN_SIZE[1], flip=False)
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])
    datasetname = config.data_mode if config.data_mode == 'MovingDroneCrowd' else config.data_mode.upper()

    # Read base scene names from the scene list file
    scene_names = []
    with open(config.scene_path, 'r') as f:
        for line in f.readlines():
            scene_names.append(line.strip('\n'))
    assert scene_names, f"No scenes found in {config.scene_path}"

    if datasetname == 'MovingDroneCrowd':
        # MDC: group sub-clips under each base scene
        per_scene_datasets = {}
        for base_scene in scene_names:
            root = os.path.join(config.dataset_path, 'frames', base_scene)
            clip_names = [c for c in os.listdir(root) if '.' not in c]
            clip_names.sort()
            assert clip_names, f"Scene {base_scene} has no sub-clip directories in {root}"
            scene_dsets = []
            for clip_name in clip_names:
                scene_clip_name = f"{base_scene}/{clip_name}"
                sub = Dataset(
                    scene_name=scene_clip_name,
                    base_path=config.dataset_path,
                    main_transform=main_transform,
                    img_transform=img_transform,
                    interval=cfg_data.VAL_FRAME_INTERVALS,
                    skip_flag=False,
                    target=True,
                    datasetname=datasetname,
                    training=training,
                )
                scene_dsets.append(sub)
            per_scene_datasets[base_scene] = scene_dsets
    else:
        # HT21 or others: each scene name is a full path, no sub-clip grouping
        per_scene_datasets = {}
        for scene in scene_names:
            sub = Dataset(
                scene_name=scene,
                base_path=config.dataset_path,
                main_transform=main_transform,
                img_transform=img_transform,
                interval=cfg_data.VAL_FRAME_INTERVALS,
                skip_flag=False,
                target=True,
                datasetname=datasetname,
                training=training,
            )
            per_scene_datasets[scene] = [sub]

    # Wrap in DataLoaders
    scene_loaders = {}
    for scene_name, dsets in per_scene_datasets.items():
        combined = dsets[0] if len(dsets) == 1 else torch.utils.data.ConcatDataset(dsets)
        assert len(combined) > 0, f"Scene {scene_name} has 0 valid samples"
        scene_loaders[scene_name] = DataLoader(
            combined, shuffle=config.shuffle, batch_size=1, drop_last=False,
            num_workers=0, collate_fn=datasets.p2r_collate_fn,
        )
    return scene_loaders