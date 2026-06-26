#!/usr/bin/env python
"""
Gate Fisher Analysis — 测量源模型各 gate 对 data loss 的敏感性。

流程: 加载 sdnet.pth → 加 identity gate → 遍历全测试集 →
      每样本算 data MSE loss → backward → 汇总 Fisher = E[grad²].
"""
import os, sys, math, json
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import import_module
from easydict import EasyDict as edict
import datasets
from datasets.dataset import P2RDataset
from datasets.utils import get_testset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from model.VIC import Video_Counter
from model_assembler import add_gates_to_attention, add_gates_to_conv
from model_assembler import GatedConv
from model.fisher import compute_fisher

# FisherGatedConv — 和 GatedConv 一样，但 forward 没有 torch.no_grad()
# 这样才能让梯度反向传播到前面的层
# class FisherGatedConv(GatedConv):
#     def forward(self, x):
#         def conv_forward(input):
#             return self.conv(input)   # 不需要 no_grad，让输入梯度正常生成

#     # 3. 用 checkpoint 包裹卷积调用
#         y = checkpoint.checkpoint(conv_forward, x)
#         # y = self.conv(x)  # 无 torch.no_grad()，梯度可流过 conv 操作
#         if self.mode == 'independent':
#             g = 2 * F.sigmoid(self.gate.view(1, -1, 1, 1))
#             return y * g
#         else:  # correlated
#             return self.gate_conv(y)

# def add_fisher_gates_to_conv(model):
#     """递归替换 Conv2d → FisherGatedConv（保留梯度流通）"""
#     for name, child in list(model.named_children()):
#         if isinstance(child, nn.Conv2d):
#             g = FisherGatedConv(child, mode='independent')
#             setattr(model, name, g)
#         elif len(list(child.children())):
#             add_fisher_gates_to_conv(
#                 child)

# ====== Config ======
DATASET_NAME = 'MovingDroneCrowd'
DATASET_PATH = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd'
SCENE_PATH   = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd/test.txt'
MODEL_PATH   = '/home/mscs/houminqiu2/SFSDNet/sdnet.pth'
OUTPUT_DIR   = '.'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg.encoder = 'VGG16_FPN'
cfg.cross_attn_embed_dim = 256
cfg.cross_attn_num_heads = 4
cfg.mlp_ratio = 4
cfg.cross_attn_depth = 2
cfg.FEATURE_DIM = 256

cfg_data = import_module(f'datasets.setting.{DATASET_NAME}').cfg_data
cfg_data.DATA_PATH = DATASET_PATH

# ====== Model ======
print("Loading model + sdnet.pth ...")
model = Video_Counter(cfg, cfg_data)
sd = torch.load(MODEL_PATH, map_location='cpu')
sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
for p in model.parameters():
    p.requires_grad = False

print("Adding identity gates ...")
add_gates_to_conv(model)
add_gates_to_attention(model)

for n, p in model.named_parameters():
    p.requires_grad = any(k in n for k in ['.gate', '.q_gate_logit', '.k_gate_logit', '.v_gate_logit'])

gate_params = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"  Gate params: {len(gate_params)}")
for g in gate_params[:10]: print(f"    {g}")
if len(gate_params) > 10: print(f"    ... and {len(gate_params)-10} more")

model.to(DEVICE).eval()

# ====== Dataset ======
print("\nLoading test set ...")
tc = edict(data_mode=DATASET_NAME, dataset_path=DATASET_PATH,
           scene_path=SCENE_PATH, partial=None, shuffle=False)
loader, ds = get_testset(tc, P2RDataset, cfg_data, training=False)
print(f"  Samples: {len(ds)}")

# ====== Module classifier ======
MOD_PREFIXES = [
    ('extractor',        'Extractor.'),
    ('head',             ('global_decoder.', 'share_decoder.', 'in_out_decoder.')),
    ('feature_fuser',    'feature_fuse.'),
    ('cross_attention',  'share_cross_attention.'),
]
def which_module(pname):
    for mod, prefixes in MOD_PREFIXES:
        if isinstance(prefixes, str): prefixes = (prefixes,)
        if any(pname.startswith(p) for p in prefixes):
            return mod
    return 'other'

# ====== Compute correct Fisher via compute_fisher ======
gate_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
print(f"  Gate params: {len(gate_params)}")

def forward_fn(batch):
    imgs, _, labels, _ = batch
    imgs = imgs.to(DEVICE)
    pg, gg, ps, gs, pio, gio, _ = model(imgs, labels)
    mu = torch.cat([pg, ps, pio], dim=1)
    target = torch.cat([gg, gs, gio], dim=1).detach()
    return mu, target

fisher, grad_mean = compute_fisher(forward_fn, loader, gate_params, DEVICE, mc_iters=20)

# ====== Build DataFrame ======
rows = []
for nm in fisher:
    mod = which_module(nm)
    f_t = fisher[nm]
    g_t = grad_mean[nm]
    if nm.endswith('.gate'):
        for c in range(f_t.shape[0]):
            rows.append({'gate': f'{nm}[{c}]', 'module': mod,
                         'grad_mean': g_t[c].item(), 'fisher': f_t[c].item()})
    elif nm.endswith('_gate_logit'):
        for h in range(f_t.shape[0]):
            rows.append({'gate': f'{nm}[h{h}]', 'module': mod,
                         'grad_mean': g_t[h, 0, 0].item(), 'fisher': f_t[h, 0, 0].item()})

df = pd.DataFrame(rows)
print(f"\nModule counts:\n{df['module'].value_counts().to_string()}")

# ΔL_i = -0.5 * g_i² / F_ii  (二阶近似的预期 loss 变化)
eps = 1e-12
df['delta_loss'] = -0.5 * df['grad_mean']**2 / (df['fisher'] + eps)



# ====== Plot ======
sns.set_theme(style='whitegrid')
MC = {'extractor':'#4C72B0','head':'#DD8452','feature_fuser':'#55A868','cross_attention':'#C44E52'}
ORDER = [m for m in ['extractor','head','feature_fuser','cross_attention'] if m in df['module'].values]


# 1 — Combined violin (raw values: Fisher raw + ΔL raw)
def plot_combined_violin(fname='gate_fisher_combined_violin.png'):
    n_mods = len(ORDER)
    fig, axes = plt.subplots(2, n_mods, figsize=(5 * n_mods, 7), sharex='col')
    if n_mods == 1:
        axes = axes.reshape(2, 1)
    metrics = [
        ('fisher', 'Fisher'),
        ('delta_loss', 'ΔL'),
    ]
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for i, mod in enumerate(ORDER):
            sub = df[df['module'] == mod]
            vals = sub[metric].values  # raw, no log
            vp = axes[row_idx][i].violinplot(vals, [0], showmeans=True, showmedians=True)
            for pc in vp['bodies']:
                pc.set_facecolor(MC[mod])
                pc.set_alpha(.7)
            vp['cmeans'].set_color('k')
            vp['cmedians'].set_color('red')
            axes[row_idx][i].set_title(f'{mod}  ({len(sub)})', fontsize=11)
            axes[row_idx][i].set_xticks([])
            axes[row_idx][i].grid(axis='y', alpha=.3)
            if i == 0:
                axes[row_idx][i].set_ylabel(ylabel)
    fig.suptitle('Gate Fisher & Importance (raw)', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {fname}')
plot_combined_violin()

# 1b — Log-scale violin (log₁₀(Fisher) + log₁₀(1 / -ΔL))
def plot_logscale_violin(fname='gate_fisher_logscale_violin.png'):
    eps = 1e-30
    n_mods = len(ORDER)
    fig, axes = plt.subplots(2, n_mods, figsize=(5 * n_mods, 7), sharex='col')
    if n_mods == 1:
        axes = axes.reshape(2, 1)
    metrics = [
        ('fisher',     'log₁₀ Fisher'),
        ('delta_loss', 'log₁₀(|ΔL|)'),
    ]
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for i, mod in enumerate(ORDER):
            sub = df[df['module'] == mod].copy()
            if metric == 'fisher':
                vals = np.log10(np.maximum(sub['fisher'].values, eps))
            else:  # delta_loss: ΔL < 0 always → -ΔL > 0 → 1/(-ΔL) → log₁₀
                vals = np.log10(np.maximum(-sub['delta_loss'].values, eps))
            vp = axes[row_idx][i].violinplot(vals, [0], showmeans=True, showmedians=True)
            for pc in vp['bodies']:
                pc.set_facecolor(MC[mod])
                pc.set_alpha(.7)
            vp['cmeans'].set_color('k')
            vp['cmedians'].set_color('red')
            axes[row_idx][i].set_title(f'{mod}  ({len(sub)})', fontsize=11)
            axes[row_idx][i].set_xticks([])
            axes[row_idx][i].grid(axis='y', alpha=.3)
            if i == 0:
                axes[row_idx][i].set_ylabel(ylabel)
    fig.suptitle('Gate Fisher & Importance (log scale)', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {fname}')
plot_logscale_violin()

# 2 — Scatter: Gradient (z-score) vs ΔL (z-score)
g_abs = df['grad_mean'].values
d_abs = df['delta_loss'].values
g_mean, g_std = g_abs.mean(), g_abs.std()
d_mean, d_std = d_abs.mean(), d_abs.std()
fig, ax = plt.subplots(figsize=(10,7))
for mod in ORDER:
    sub = df[df['module']==mod]
    ax.scatter((sub['grad_mean'] - g_mean) / g_std,
               (sub['delta_loss'] - d_mean) / d_std,
               c=MC[mod], label=mod, alpha=.5, s=8)
ax.axhline(0, c='gray', ls='--', lw=.5); ax.axvline(0, c='gray', ls='--', lw=.5)
ax.set_xlabel('Gradient (z-score)'); ax.set_ylabel('ΔL (z-score)')
ax.set_title('Gradient vs ΔL (colored by module)'); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'gate_fisher_scatter.png'), dpi=150, bbox_inches='tight'); plt.close()
print('  → gate_fisher_scatter.png')

print(f"\n✓ All done.  {df.shape[0]} gates across {df['module'].nunique()} modules.")
