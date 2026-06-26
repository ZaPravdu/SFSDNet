#!/usr/bin/env python
"""
梯度流诊断：检查梯度从 loss 一路反向传播到各层的状况。
打印关键位置的梯度统计，找出梯度在哪里消失了。
"""
import os, sys
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from importlib import import_module
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from model.VIC import Video_Counter
from model_assembler import add_gates_to_conv, add_gates_to_attention, GatedConv
from model.gates import BaseGatedModule
import datasets
from datasets.dataset import P2RDataset
from datasets.utils import get_testset

DATASET_NAME = 'MovingDroneCrowd'
DATASET_PATH = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd'
SCENE_PATH   = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd/test.txt'
MODEL_PATH   = '/home/mscs/houminqiu2/SFSDNet/sdnet.pth'
DEVICE       = 'cuda'

cfg.encoder = 'VGG16_FPN'
cfg.cross_attn_embed_dim = 256
cfg.cross_attn_num_heads = 4
cfg.mlp_ratio = 4
cfg.cross_attn_depth = 2
cfg.FEATURE_DIM = 256
cfg_data = import_module(f'datasets.setting.{DATASET_NAME}').cfg_data
cfg_data.DATA_PATH = DATASET_PATH

model = Video_Counter(cfg, cfg_data)
sd = torch.load(MODEL_PATH, map_location='cpu')
sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
for p in model.parameters():
    p.requires_grad = False
add_gates_to_conv(model)
for n, p in model.named_parameters():
    p.requires_grad = any(k in n for k in ['.gate'])
model.to(DEVICE).eval()

# === Gradient hook 系统：在每个 GatedConv 后收集 gate.grad + 输入/输出统计 ===
grad_stats = {}  # name → {'gate_grad_norm': [], 'output_alive_frac': [], 'output_mean_abs': []}

def make_hooks(name, gconv):
    """给 GatedConv.gate 注册 backward hook, 收集进入 gate 的梯度"""
    def bw_hook(grad):
        if name not in grad_stats:
            grad_stats[name] = {'gate_grad_norm': [], 'gate_grad_absmean': []}
        grad_stats[name]['gate_grad_norm'].append(grad.norm().item())
        grad_stats[name]['gate_grad_absmean'].append(grad.abs().mean().item())
    gconv.gate.register_hook(bw_hook)

for n, m in model.named_modules():
    if isinstance(m, GatedConv):
        make_hooks(n, m)

# === 数据 ===
tc = edict(data_mode=DATASET_NAME, dataset_path=DATASET_PATH,
           scene_path=SCENE_PATH, partial=None, shuffle=False)
loader, ds = get_testset(tc, P2RDataset, cfg_data, training=False)

# 跑 10 个 batch
for bidx, batch in enumerate(tqdm(loader, desc='Grad flow', total=10)):
    if bidx >= 10:
        break
    imgs, _, labels, _ = batch
    imgs = imgs.to(DEVICE)
    model.zero_grad()
    pg, gg, ps, gs, pio, gio, _ = model(imgs, labels)
    d = cfg_data.DEN_FACTOR
    loss = (F.mse_loss(pg*d, gg*d) + 10*F.mse_loss(ps*d, gs*d) + F.mse_loss(pio*d, gio*d)) / 3
    loss.backward()

# 打印梯度统计（按模块分组）
print("\n" + "=" * 85)
print(f"{'Gate location':55s} {'grad_norm':>10s} {'grad_absmean':>12s}  {'dead(0)':>7s}")
print("-" * 85)
MOD_ORDER = [
    'Extractor.layer1', 'Extractor.layer2', 'Extractor.layer3',
    'Extractor.neck2f', 'Extractor.feature_head',
    'global_decoder', 'share_decoder', 'in_out_decoder',
    'feature_fuse',
]
MOD_GROUP = {}
for k in grad_stats:
    mod = 'other'
    for prefix in MOD_ORDER:
        if k.startswith(prefix):
            mod = prefix
            break
    MOD_GROUP.setdefault(mod, []).append(k)

zero_gates = []
for mod in MOD_ORDER + ['other']:
    if mod not in MOD_GROUP:
        continue
    gates = MOD_GROUP[mod]
    for gname in sorted(gates):
        stats = grad_stats[gname]
        gn = np.mean(stats['gate_grad_norm'])
        ga = np.mean(stats['gate_grad_absmean'])
        gm = np.min(stats['gate_grad_norm'])
        is_zero = gm == 0 and gn == 0
        if is_zero:
            zero_gates.append(gname)
        print(f"{gname:55s} {gn:10.2e} {ga:12.2e}  {'⚫ DEAD' if is_zero else '✓':>7s}")

# 按模块统计 zero 比例
print("\n\n" + "=" * 85)
print("Zero-gradient gate summary by module")
print("-" * 85)
for mod in MOD_ORDER:
    if mod not in MOD_GROUP:
        continue
    gates = MOD_GROUP[mod]
    n_zero = sum(1 for g in gates if g in zero_gates)
    n_total = len(gates)
    if n_total > 0:
        print(f"  {mod:45s} {n_zero:5d} / {n_total:5d}  ({100*n_zero/n_total:.1f}%)")

C_gates = sum(1 for g in grad_stats if '.gate' in g and not any(a in g for a in ['logit']))
C_zero = len(zero_gates)
print(f"\n  {'TOTAL':45s} {C_zero:5d} / {C_gates:5d}  ({100*C_zero/max(C_gates,1):.1f}%)")
print(f"\nZero gates list ({len(zero_gates)} total):")
for g in zero_gates:
    print(f"  {g}")

print("\n✓ Done")
