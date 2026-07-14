#!/usr/bin/env python
"""
深度梯度流诊断：追踪每个 GatedConv gate[i] 的通道级梯度。
找出哪些通道的梯度为零，并分析原因。
"""
import os, sys
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import defaultdict
from importlib import import_module
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from model.VIC import Video_Counter
from model_assembler import add_gates_to_conv, add_gates_to_attention, GatedConv
from model.gates import BaseGatedModule, GatedAttention, GatedCrossAttention
import datasets
from datasets.dataset import TTDADataset
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
# add_gates_to_attention(model)  # 注意力门控先不加入
for n, p in model.named_parameters():
    p.requires_grad = any(k in n for k in ['.gate'])
model.to(DEVICE).eval()

# === 通道级 Hook：收集每个 gate[i] 的梯度绝对值 ===
# {name: {channel: [batch_grad]}}
chan_grads = defaultdict(lambda: defaultdict(list))

def make_channel_hook(name, gconv):
    """为每个 gate[i] 注册 hook"""
    def bw_hook(grad):
        g = grad.detach().cpu().float()  # shape (C,)
        for c in range(g.shape[0]):
            chan_grads[name][c].append(g[c].item())
    gconv.gate.register_hook(bw_hook)

for n, m in model.named_modules():
    if isinstance(m, GatedConv):
        make_channel_hook(n, m)

# === Focused gradient path tracing ===
# 在关键位置注册 hook 看梯度大小
intermediate_grads = {}

def hook_fn(name):
    def fn(grad):
        intermediate_grads[name] = grad.detach().cpu().float().abs().mean().item()
    return fn

# 在 Extractor 的 backbone 输出处注册 hook
hooks = []
# x1, x2, x3 分别是 VGG layer1, layer2, layer3 的输出
# 我们 hook Extractor 内部的中间层
for name, mod in model.Extractor.named_modules():
    if name == 'layer1' and hasattr(mod, 'register_backward_hook'):
        hooks.append(mod.register_backward_hook(lambda mod, gin, gout:
            setattr(intermediate_grads, 'layer1_grad',
                    (gin[0].detach().cpu().float().abs().mean().item() if gin[0] is not None else 0))))
    # 简化的方法：hook features[-1] 和 features[0] 这些
    # 通过 VIC.forward 的输出会自动跟踪

# === 数据 ===
tc = edict(data_mode=DATASET_NAME, dataset_path=DATASET_PATH,
           scene_path=SCENE_PATH, partial=None, shuffle=False)
loader, ds = get_testset(tc, TTDADataset, cfg_data, training=False)

N_BATCHES = 20
for bidx, batch in enumerate(tqdm(loader, desc='Grad flow', total=N_BATCHES)):
    if bidx >= N_BATCHES:
        break
    imgs, _, labels, _ = batch
    imgs = imgs.to(DEVICE)
    model.zero_grad()
    pg, gg, ps, gs, pio, gio, _ = model(imgs, labels)
    d = cfg_data.DEN_FACTOR
    loss = (F.mse_loss(pg*d, gg*d) + 10*F.mse_loss(ps*d, gs*d) + F.mse_loss(pio*d, gio*d)) / 3
    loss.backward()

for h in hooks:
    h.remove()

# === 分析 ===
print("\n" + "=" * 90)
print("Channel-level gradient analysis")
print("=" * 90)

# 按模块分组并打印每个模块中零梯度通道的比例
MOD_PREFIXES = [
    ('Extractor.layer1', 'layer1'), ('Extractor.layer2', 'layer2'),
    ('Extractor.layer3', 'layer3'), ('Extractor.neck2f', 'neck2f'),
    ('Extractor.feature_head', 'feature_head'),
    ('global_decoder', 'g_dec'), ('share_decoder', 's_dec'),
    ('in_out_decoder', 'io_dec'), ('feature_fuse', 'fuse'),
]

def classify(name):
    for prefix, label in MOD_PREFIXES:
        if name.startswith(prefix):
            return label
    return 'other'

all_results = []
for gate_name in sorted(chan_grads.keys()):
    mod_label = classify(gate_name)
    ch_dict = chan_grads[gate_name]
    C = len(ch_dict)
    zero_ch = 0
    total_ch = 0
    for c in range(C):
        vals = ch_dict[c]
        if not vals:
            continue
        total_ch += 1
        grad_abs_mean = np.mean([abs(v) for v in vals])
        if grad_abs_mean == 0.0:
            zero_ch += 1
    all_results.append((gate_name, mod_label, zero_ch, total_ch))

# 按模块汇总
mod_stats = defaultdict(lambda: {'zero': 0, 'total': 0})
for name, mod, z, t in all_results:
    mod_stats[mod]['zero'] += z
    mod_stats[mod]['total'] += t

print(f"\n{'Module':30s} {'Zero-Ch':>8s} {'Total-Ch':>10s} {'%':>8s}")
print("-" * 60)
for mod in ['layer1', 'layer2', 'layer3', 'neck2f', 'feature_head',
            'g_dec', 's_dec', 'io_dec', 'fuse', 'other']:
    s = mod_stats[mod]
    if s['total'] > 0:
        pct = 100 * s['zero'] / s['total']
        bar = '#' * int(pct / 5) + ' ' * (20 - int(pct / 5))
        print(f"{mod:30s} {s['zero']:8d} {s['total']:10d} {pct:7.1f}%  [{bar}]")

total_z = sum(s['zero'] for s in mod_stats.values())
total_t = sum(s['total'] for s in mod_stats.values())
print("-" * 60)
print(f"{'TOTAL':30s} {total_z:8d} {total_t:10d} {100*total_z/max(total_t,1):7.1f}%")

# === 检查 conv 输出的绝对值大小（并非是否为0，而是量级）===
print("\n\n" + "=" * 90)
print("Conv output magnitude analysis (why gradients vanish?)")
print("=" * 90)

# 前向一次收集 conv 输出统计
conv_out_stats = {}
def conv_hook_fn(name):
    def fn(m, i, o):
        o = o.detach().cpu().float()
        B, C, H, W = o.shape
        for c in range(C):
            chan_out = o[:, c, :, :]
            key = f"{name}[{c}]"
            if key not in conv_out_stats:
                conv_out_stats[key] = []
            conv_out_stats[key].append({
                'mean': chan_out.mean().item(),
                'absmean': chan_out.abs().mean().item(),
                'frac_nonzero': (chan_out != 0).float().mean().item(),
                'frac_positive': (chan_out > 0).float().mean().item(),
            })
    return fn

conv_hooks = []
for n, m in model.named_modules():
    if isinstance(m, GatedConv):
        conv_hooks.append(m.conv.register_forward_hook(conv_hook_fn(n + '.conv')))

# 一个 batch 就够了
model.zero_grad()
imgs, _, labels, _ = next(iter(loader))
imgs = imgs.to(DEVICE)
with torch.no_grad():
    model(imgs, labels)
for h in conv_hooks:
    h.remove()

# 检查 Extractor.layer1.0 的 conv 输出——所有通道的输出量级
print("\nExtractor.layer1.0 (first conv) output stats per channel:")
print(f"{'Channel':>8s} {'absmean':>10s} {'frac>0':>8s} {'nonzero':>8s}")
layer1_0_keys = sorted([k for k in conv_out_stats if 'Extractor.layer1.0' in k])
for k in layer1_0_keys[:64]:  # all 64 channels
    stats_list = conv_out_stats[k]
    if stats_list:
        s = stats_list[0]
        print(f"{k.rsplit('[', 1)[-1].rstrip(']'):>8s} {s['absmean']:10.3e} {s['frac_positive']:8.4f} {s['frac_nonzero']:8.4f}")

# 整体数值分布
print("\nConv output absmean distribution (all Extractor GatedConvs):")
all_absmeans = [s[0]['absmean'] for k, s in conv_out_stats.items()
                if 'Extractor.' in k and s]
if all_absmeans:
    all_absmeans = np.array(all_absmeans)
    print(f"  min:    {all_absmeans.min():.3e}")
    print(f"  median: {np.median(all_absmeans):.3e}")
    print(f"  mean:   {all_absmeans.mean():.3e}")
    print(f"  max:    {all_absmeans.max():.3e}")
    print(f"  zeros:  {(all_absmeans == 0).sum()} / {len(all_absmeans)}")

print("\n✓ Done")
