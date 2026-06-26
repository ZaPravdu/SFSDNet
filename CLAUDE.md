# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SFSDNet** — video crowd counting with scene-consistent pseudo-labeling. The model takes pairs of video frames and predicts three density maps per frame: **global** (all people), **shared** (people visible in both frames), and **in/out** (people entering/exiting). It uses a teacher-student framework (P2R training) for semi-supervised learning on unlabeled data, plus gate-based regularization (L1, L2, or KL). Supported datasets: MovingDroneCrowd, HT21, SENSE, UAVVIC.

## Commands

### Training

```bash
# Supervised training (original trainer, distributed)
python train.py

# Semi-supervised P2R training (PyTorch Lightning)
python train_p2r.py

# Simplified supervised training (PyTorch Lightning, uses train_script.py)
python train_script.py
```

### Testing / Inference

```bash
# Full test with metrics (MAE, MSE, WRAE, MIAE, MOAE)
python test.py --DATASET MovingDroneCrowd --model_path <path> --GPU_ID 0

# Consistency regularization inference (generates pseudo-density maps with uncertainty masks)
python inference.py

# Single sample test
python single_sample_test.py

# Inference with VGGAE
python VGGAE_inference.py
```

### Analysis & Visualization

```bash
# Correlation analysis between uncertainty and error
python analyze.py

# Error map generation from pseudo-density maps
python generate_error_mask.py

# Generate pseudo-density maps from pretrained model
python generate_pseudo_density_map.py

# Visualization of results
python visualize.py

# Data visualization / correlation analysis
python data_visualization.py

# Analyze per-scene errors
python analyze_error.py
```

### Condor (HTCondor cluster)

```bash
# Submit training job to HTCondor
condor_submit submit_job.condor
```

### Configuration

Edit [config.py](config.py) to set:
- `cfg.DATASET` — dataset selection (MovingDroneCrowd, HT21, SENSE)
- `cfg.encoder` — backbone (VGG16_FPN, PCPVT, ResNet_50_FPN)
- `cfg.GPU_ID` — which GPUs to use
- `cfg.LR_Base`, `cfg.MAX_EPOCH`, etc. — training hyperparameters

For P2R training, modify the `TrainConfig` class in [train_p2r.py](train_p2r.py) to adjust freezing, regularization modes, and experiment naming.

## Architecture

### Model Pipeline

```
Input: (img0, img1)  — two video frames
  │
  ├── Backbone Extractor (VGG16_FPN / ResNet_50_FPN / PCPVT)
  │     └── Multi-scale pyramid features
  │
  ├── Cross-Attention Module (3 levels × N depths)
  │     └── Frame-to-frame bidirectional cross-attention → shared features
  │
  ├── FeatureFusionModule
  │     └── Fuses attention output with backbone features across levels
  │
  ├── GlobalDecoder → global density map (all people)
  ├── ShareDecoder → share density map (people visible in both frames)
  └── InOutDecoder → in/out density map (people entering/exiting)
        └── Input = global - share (residual)
```

### Key Model Classes

- **`Video_Counter`** ([model/VIC.py](model/VIC.py)) — Core model: backbone + cross-attention + three decoders. `forward()` takes (img, target) and returns all three density maps (pre_global, gt_global, pre_share, gt_share, pre_in_out, gt_in_out, loss_dict).

- **`SFSDNet`** ([model_assembler.py](model_assembler.py)) — LightningModule wrapping `Video_Counter` for supervised training with pseudo-density map loss. Adds MSE loss against both GT density maps and pre-computed pseudo-density maps (from a pretrained SDNet). Freezing controls: `freeze_backbone`, `freeze_head`, `freeze_feature_fuse`.

- **`P2RModel`** ([model_assembler.py](model_assembler.py)) — LightningModule for semi-supervised P2R training. Teacher-student architecture:
  - Two training modes: `supervised` (uses GT density maps) or `p2r` (teacher generates pseudo-labels on weak augmentations, student learns from strong augmentations)
  - Optional EMA update of teacher from student
  - Optional density map reconstruction (extract points → re-blur)
  - Gate regularization: `GatedConv`, `GatedAttention`, `GatedCrossAttention` modules inserted into the student network
  - Regularization modes: `kl` (variational inference), `l2`, `l1` on gate parameters

- **`HyperModel`** ([model_assembler.py](model_assembler.py)) — Base LightningModule with AdamW + CosineAnnealingLR.

- **`GatedConv`** / **`GatedAttention`** / **`GatedCrossAttention`** ([model_assembler.py](model_assembler.py)) — Drop-in replacements for Conv2d / Attention / CrossAttention that add per-channel or per-head gating with L1/L2/KL regularization. Gates start at identity (gate=1.0) and are learnable.

- **`add_gates_to_conv()`** / **`add_gates_to_attention()`** ([model_assembler.py](model_assembler.py)) — Recursively replace all Conv2d/Attention/CrossAttention modules with gated versions.

### Decoders ([model/decoder.py](model/decoder.py))

All three decoders are 4-layer Conv2d stacks with GroupNorm, ReLU, and bilinear upsampling (2× per layer, total 16× upsampling from feature map to full resolution). `GlobalDecoder` and `ShareDecoder` are architecturally identical; `InOutDecoder` is a single-stage 4-conv stack without upsampling.

### Datasets ([datasets/dataset.py](datasets/dataset.py))

- **`Dataset`** (training) — Loads consecutive frame pairs from a scene. Computes `share_mask0`, `share_mask1`, `outflow_mask`, `inflow_mask` based on person_id matching or precomputed in/out labels.
- **`TestDataset`** (test/val) — Loads frame pairs at configurable intervals. Supports `skip_flag` for interval-based sampling.
- **`P2RDataset`** — Extends `TestDataset` with strong augmentation pipeline (ColorJitter, Grayscale, GaussianBlur) for P2R training.
- **Collation**: `collate_fn` for supervised batches, `p2r_collate_fn` for (weak, strong, labels) tuples.

Dataset-specific path/annotation loaders:
- `MDC_ImgPath_and_Target` — MovingDroneCrowd CSV annotations
- `UAVVIC_ImgPath_and_Target` — UAVVIC JSONL annotations
- `HT21_ImgPath_and_Target` — HT21 MOT format annotations
- `SENSE_ImgPath_and_Target` — SENSE custom label format

### Dataset Configurations ([datasets/setting/](datasets/setting/))

Each dataset has a settings file exposing `cfg_data` with:
- `TRAIN_SIZE`, `DATA_PATH`, mean/std normalization
- Frame intervals, batch sizes, density factor (`DEN_FACTOR=200`)
- Train/val/test list filenames

### Inference Pipeline ([inference/engine.py](inference/engine.py))

The `PseudoInference` class runs the model, applies data augmentation transforms, estimates patch-wise uncertainty via std over multiple augmentations, and generates uncertainty-weighted pseudo-density maps with masks. Output: concatenated [global, share, in_out, mask] .npy files.

### Training Entry Points

1. **`train.py`** (legacy) — Manual training loop with distributed support (DistributedDataParallel), Adam optimizer, per-epoch validation with scene-level metrics (MAE, MSE, WRAE, MIAE, MOAE). Uses `train_resize_transform` with center-crop-around-centroid.

2. **`train_script.py`** (simplified Lightning) — Quick supervised training loop. Wraps `SFSDNet` in a PyTorch Lightning `Trainer` with WandbLogger, ModelCheckpoint, EarlyStopping.

3. **`train_p2r.py`** (P2R Lightning) — Teacher-student semi-supervised training. Uses `P2RModel` and `P2RDataset` with strong/weak augmentation pairs.

## Workflow Guidelines

### General
- Prefer editing over rewriting an entire file
- Do not re-read a file unless it has been edited since last read
- Keep output concise, but maintain thorough reasoning in internal deliberation

### Code Style
- Nesting depth must not exceed 4 levels

### Key Metrics

- **MAE / MSE** — Mean Absolute/MSE error per-frame (global counting)
- **WRAE** — Weighted Relative Absolute Error across scenes
- **MIAE / MOAE** — Mean Inflow/Outflow Absolute Error
- **Seq_MAE** — Sequence-level MAE (scene-level total count error)
- **Correlation analysis** — Pearson correlation between model uncertainty (patch std) and prediction error (MAE/MSE), computed per scene and per density channel (global, share, io)
