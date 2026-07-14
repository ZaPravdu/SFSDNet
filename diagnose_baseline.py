"""
基线诊断：加载 Video_Counter + 预训练权重，不注入 gate，不修改任何参数。
跑一个 batch 看原始模型输出量级是否正确。
"""
import os, sys
import torch, torch.nn.functional as F
import numpy as np
from importlib import import_module
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from model.VIC import Video_Counter
import datasets
from datasets.dataset import TTDADataset
from datasets.utils import get_testset

# ── 配置 ──────────────────────────────────────────────────────────
DATASET_NAME = 'MovingDroneCrowd'
DATASET_PATH = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd'
SCENE_PATH   = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd/test.txt'
MODEL_PATH   = '/home/mscs/houminqiu2/SFSDNet/sdnet.pth'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
SINGLE_SCENE = 'scene_25'   # 指定单场景时改为具体场景名，否则用 None

# ── 模型 ──────────────────────────────────────────────────────────
cfg.encoder = 'VGG16_FPN'
cfg.cross_attn_embed_dim = 256
cfg.cross_attn_num_heads = 4
cfg.mlp_ratio = 4
cfg.cross_attn_depth = 2
cfg.FEATURE_DIM = 256

cfg_data = import_module(f'datasets.setting.{DATASET_NAME}').cfg_data
cfg_data.DATA_PATH = DATASET_PATH

print(f"Building Video_Counter (encoder={cfg.encoder})...")
model = Video_Counter(cfg, cfg_data).to(DEVICE)
model.eval()

sd = torch.load(MODEL_PATH, map_location='cpu')
sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print(f"  Missing keys: {missing}")
if unexpected:
    print(f"  Unexpected keys: {unexpected}")
print(f"  Weights loaded from {MODEL_PATH}")

# ── 数据 ──────────────────────────────────────────────────────────
tc = edict(
    data_mode=DATASET_NAME,
    dataset_path=DATASET_PATH,
    scene_path=SCENE_PATH,
    partial=None,
    shuffle=False,
)
if SINGLE_SCENE:
    tc.single_scene = SINGLE_SCENE

loader, ds = get_testset(tc, TTDADataset, cfg_data, training=False)
print(f"Dataset: {len(ds)} samples")

# ── 跑一个 batch ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE: Video_Counter raw output (no gates, no modifications)")
print("=" * 70)

for bidx, batch in enumerate(loader):
    if bidx >= 1:
        break

    imgs, _, labels = batch
    imgs = imgs.to(DEVICE)
    print(f"\nBatch {bidx}: images shape={list(imgs.shape)}, frames={imgs.size(0)}")

    with torch.no_grad():
        pre_global, gt_global, pre_share, gt_share, pre_io, gt_io, _ = model(imgs, labels)

    d = cfg_data.DEN_FACTOR
    print(f"\n  DEN_FACTOR = {d}")
    print(f"\n  ── Global density map ──")
    print(f"    pre_global.sum()    = {pre_global.sum().item():.2f}")
    print(f"    gt_global.sum()     = {gt_global.sum().item():.2f}")
    print(f"    decoder_raw_sum     = {(pre_global * d).sum().item():.2f}  (= pre * {d})")
    print(f"    gt_scaled_sum       = {(gt_global * d).sum().item():.2f}  (= gt * {d})")
    print(f"    decoder/gt_ratio    = {(pre_global * d).sum() / (gt_global * d).sum():.4f}")

    print(f"\n  ── Per-frame stats ──")
    for i in range(len(imgs)):
        n_people = len(labels[i]['points']) if 'points' in labels[i] else 0
        gt_sum_i = gt_global[i].sum().item()
        pre_sum_i = pre_global[i].sum().item()
        ratio_ppl = gt_sum_i / n_people if n_people > 0 else None
        print(f"    frame[{i}]: pre={pre_sum_i:.2f}, gt={gt_sum_i:.2f}, "
              f"people={n_people}, gt/people={ratio_ppl}")

    print(f"\n  ── Share density map ──")
    print(f"    pre_share.sum()     = {pre_share.sum().item():.2f}")
    print(f"    gt_share.sum()      = {gt_share.sum().item():.2f}")

    print(f"\n  ── In/Out density map ──")
    print(f"    pre_io.sum()        = {pre_io.sum().item():.2f}")
    print(f"    gt_io.sum()         = {gt_io.sum().item():.2f}")

    # 计算 loss（和训练时一致）
    global_loss = F.mse_loss(pre_global * d, gt_global * d)
    share_loss = F.mse_loss(pre_share * d, gt_share * d)
    io_loss = F.mse_loss(pre_io * d, gt_io * d)
    total_loss = (global_loss + 10 * share_loss + io_loss) / 3
    print(f"\n  ── Loss (same as training) ──")
    print(f"    gt_global_loss      = {global_loss.item():.6f}")
    print(f"    gt_share_loss       = {share_loss.item():.6f}")
    print(f"    gt_in_out_loss      = {io_loss.item():.6f}")
    print(f"    total_loss          = {total_loss.item():.6f}")

    print(f"\n  ── Unscaled MSE (test/global_gt_loss equivalent) ──")
    print(f"    MSE(pre_global, gt_global) = {F.mse_loss(pre_global, gt_global).item():.8f}")

print("\n✓ Baseline complete")
