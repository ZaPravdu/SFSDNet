import json
from importlib import import_module

import torch
from tqdm import tqdm

from config import cfg
import datasets
import model_assembler
from model.VIC import Video_Counter

# load model

state = torch.load('./ep_120_iter_105000_mae_10.119_mse_13.722_seq_MAE_29.751_WRAE_25.237_MIAE_3.157_MOAE_2.663.pth')
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
# load data
scenes, restore_transform = datasets.loading_testset(data_mode, 4, False, mode='test')


def calculate_mae(output: torch.Tensor, target: torch.Tensor):
    error = (target-output).abs()
    return error


def calculate_mse(output: torch.Tensor, target: torch.Tensor):
    error = (target-output)**2
    return error

sample_errors = {}
# inference
for scene in scenes:
    test_loader = scene[1]
    frame_output = {}
    for i, data in enumerate(tqdm(test_loader)):
        images, targets = data
        images = images
        pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss = model(images, targets)

        density_mae = calculate_mae(pre_global_den, gt_global_den)
        density_mse = calculate_mse(pre_global_den, gt_global_den)

        for i, target in enumerate(targets, 0):
            frame_output[target['frame']] = {'dens_mae': density_mae[i].detach().cpu().numpy(), 'dens_mse': density_mse[i].detach().cpu().numpy()}

    sample_errors[scene[0]] = frame_output
    break

ae_path = 'weight/VIC/VGGAE/epoch=03-val_loss=0.7279.ckpt'
model = model_assembler.VGGAE.load_from_checkpoint(checkpoint_path=ae_path).cuda()
model.eval()

for scene in scenes:
    test_loader = scene[1]
    for i, data in enumerate(tqdm(test_loader)):
        images, targets = data
        images = images.cuda()
        recon = model(images)

        recon_mae = calculate_mae(recon, images)
        recon_mse = calculate_mse(recon, images)

        for i, target in enumerate(targets, 0):
            sample_errors[scene[0]][target['frame']]['recon_mae'] = recon_mae[i].detach().cpu().numpy()
            sample_errors[scene[0]][target['frame']]['recon_mse'] = recon_mse[i].detach().cpu().numpy()
    break

with open('test_error.json', "w") as f:
    json.dump(sample_errors, f)

# calculate metrics for density map

# calculate metrics for reconstruction

# calculate correlation factor

