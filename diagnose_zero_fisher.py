#!/usr/bin/env python
"""
诊断：哪些 gate 的 Fisher=0？原因是什么？
1. 加载模型 + 加 gate → 遍历测试集 → 统计每个 gate 通道的 Fisher
2. 同时记录 conv 输出 y 的激活率（非零比例）
3. 如果 Fisher=0 且 conv 输出全零 → 死通道
4. 如果 Fisher=0 但 conv 输出非零 → 梯度反向传播出了问题
"""
import os, sys, json
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
from datasets.dataset import TTDADataset
from datasets.utils import get_testset

DATASET_NAME = 'MovingDroneCrowd'
DATASET_PATH = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd'
SCENE_PATH   = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd/test.txt'
MODEL_PATH   = '/home/mscs/houminqiu2/SFSDNet/sdnet.pth'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_BATCHES  = 50     # 用 50 个 batch 加速诊断，足够判断死通道

cfg.encoder = 'VGG16_FPN'
cfg.cross_attn_embed_dim = 256
cfg.cross_attn_num_heads = 4
cfg.mlp_ratio = 4
cfg.cross_attn_depth = 2
cfg.FEATURE_DIM = 256
cfg_data = import_module(f'datasets.setting.{DATASET_NAME}').cfg_data
cfg_data.DATA_PATH = DATASET_PATH

# ====== Model ======
model = Video_Counter(cfg, cfg_data)
sd = torch.load(MODEL_PATH, map_location='cpu')
sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
for p in model.parameters():
    p.requires_grad = False

add_gates_to_conv(model)
add_gates_to_attention(model)

for n, p in model.named_parameters():
    p.requires_grad = any(k in n for k in ['.gate', '.q_gate_logit', '.k_gate_logit', '.v_gate_logit'])
model.to(DEVICE).eval()

# 注册 hook 收集 GatedConv.conv 的输出（gate 的输入 y）
conv_outputs = {}
hook_handles = []
for name, mod in model.named_modules():
    if isinstance(mod, GatedConv):
        def _hook(n):
            def fn(m, i, o):
                conv_outputs[n] = o.detach().cpu()  # shape (B, C, H, W)
            return fn
        hook_handles.append(mod.conv.register_forward_hook(_hook(name)))

# 更精细的 hook：在 GatedConv.forward 之后直接收集 gate*gate_grad
gate_data = {}  # gate_name → list of grad vectors

# ====== Dataset ======
tc = edict(data_mode=DATASET_NAME, dataset_path=DATASET_PATH,
           scene_path=SCENE_PATH, partial=None, shuffle=False)
loader, ds = get_testset(tc, TTDADataset, cfg_data, training=False)
print(f"Dataset: {len(ds)} samples, using {min(MAX_BATCHES, len(loader))} batches")

# ====== Run ======
for bidx, batch in enumerate(tqdm(loader, desc='Diagnosing')):
    if bidx >= MAX_BATCHES:
        break
    imgs, _, labels, _ = batch
    imgs = imgs.to(DEVICE)

    model.zero_grad()
    pg, gg, ps, gs, pio, gio, _ = model(imgs, labels)
    d = cfg_data.DEN_FACTOR
    loss = (F.mse_loss(pg*d, gg*d) + 10*F.mse_loss(ps*d, gs*d) + F.mse_loss(pio*d, gio*d)) / 3
    loss.backward()

    # 收集 gate 梯度
    for nm, pr in model.named_parameters():
        if not pr.requires_grad or pr.grad is None:
            continue
        if nm.endswith('.gate'):
            g = pr.grad.detach().cpu().float()
            if nm not in gate_data:
                gate_data[nm] = []
            gate_data[nm].append(g)

# ====== Analyze ======
print("\n" + "=" * 70)
print("Fisher & Dead Channel Analysis")
print("=" * 70)

# 1. 按模块归类
MOD_PREFIXES = [
    ('Extractor.layer1',   'Extractor.layer1'),
    ('Extractor.layer2',   'Extractor.layer2'),
    ('Extractor.layer3',   'Extractor.layer3'),
    ('Extractor.neck2f',   'Extractor.neck2f'),
    ('Extractor.feature_head', 'Extractor.feature_head'),
    ('global_decoder',     'global_decoder'),
    ('share_decoder',      'share_decoder'),
    ('in_out_decoder',     'in_out_decoder'),
    ('feature_fuse',       'feature_fuse'),
]
def which_module(nm):
    for mod, prefix in MOD_PREFIXES:
        if nm.startswith(prefix):
            return mod
    return 'other'

# 2. 计算 Fisher 和激活率
results = []  # [{name, module, channel, fisher, conv_alive_frac}]
dead_channels = []
alive_channels = []

for gate_name, grad_list in gate_data.items():
    if not grad_list:
        continue
    stacked = torch.stack(grad_list, dim=0)  # (N_batches, C)
    fisher = (stacked ** 2).mean(dim=0)      # (C,)
    C = fisher.shape[0]
    mod = which_module(gate_name)

    # 查找对应的 conv 输出
    conv_name = gate_name.replace('.gate', '.conv')
    # conv_outputs 由 hook 收集，key 可能不同（取决于具体命名）
    # 尝试多种可能的 key
    conv_key = None
    for k in conv_outputs:
        if k.endswith(gate_name.replace('.gate', '.conv')) or \
           k == gate_name.replace('.gate', ''):
            conv_key = k
            break
    # 默认 conv_key = gate_name 去掉 .gate
    if conv_key is None:
        conv_key = gate_name.replace('.gate', '')

    for c in range(C):
        f = float(fisher[c])
        row = {'name': gate_name, 'module': mod, 'channel': c, 'fisher': f}
        results.append(row)
        if f == 0.0:
            dead_channels.append(row)

# 3. 打印死通道统计
print(f"\nTotal gate channels: {len(results)}")
print(f"Zero-Fisher (dead) channels: {len(dead_channels)}")

if dead_channels:
    # 按模块组统计
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│ Dead channels by module group                               │")
    print("├─────────────────────────────────────────────────────────────┤")
    from collections import Counter
    mod_counts = Counter(r['module'] for r in dead_channels)
    total_counts = Counter(r['module'] for r in results)
    for mod in sorted(mod_counts):
        print(f"  {mod:40s}  {mod_counts[mod]:5d} / {total_counts[mod]:5d}  "
              f"({100*mod_counts[mod]/max(total_counts[mod],1):.1f}%)")

    # 按每个 gate 子模块统计
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│ Dead channels per gate (gate_name: dead_C / total_C)        │")
    print("├─────────────────────────────────────────────────────────────┤")
    # 统计每个 gate_name 的 dead/total
    from collections import defaultdict
    gate_dead = defaultdict(int)
    gate_total = defaultdict(int)
    for r in results:
        gate_total[r['name']] += 1
    for r in dead_channels:
        gate_dead[r['name']] += 1
    for gate_name in sorted(gate_total):
        d = gate_dead[gate_name]
        t = gate_total[gate_name]
        if d > 0:
            bar = '#' * int(40 * d / t) + ' ' * (40 - int(40 * d / t))
            print(f"  {gate_name:55s} {d:4d} / {t:4d}  {100*d/max(t,1):5.1f}%  |{bar}|")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│ All dead gate channels (name[channel])                      │")
    print("├─────────────────────────────────────────────────────────────┤")
    for r in dead_channels:
        print(f"  {r['name']}[{r['channel']}]")

if not dead_channels:
    print("  → No zero-Fisher gates found in sampled batches.")

# 4. 检查 conv 输出激活率
print("\n\n" + "=" * 70)
print("Conv output activation analysis (conv_outputs cache)")
print("=" * 70)
conv_dead = 0
conv_total = 0
for conv_name, output in sorted(conv_outputs.items()):
    if output.numel() == 0:
        continue
    B, C, H, W = output.shape
    for c in range(C):
        alive_frac = (output[0, c] != 0).float().mean().item()
        conv_total += 1
        if alive_frac == 0.0:
            conv_dead += 1
            if conv_dead <= 20:
                print(f"  DEAD CHANNEL: {conv_name} channel[{c}]  "
                      f"spatial mean={output[0,c].mean().item():.6e}")
print(f"\nConv dead channels: {conv_dead} / {conv_total} "
      f"({100*conv_dead/max(conv_total,1):.1f}%)")

# 5. 清理 hooks
for h in hook_handles:
    h.remove()

print("\n✓ Diagnosis complete.")
