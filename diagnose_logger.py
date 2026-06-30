"""
诊断日志模块：在训练关键路径收集指标，输出 JSON 报告。
覆盖五个层面：配置 / 模型 / 数据 / 梯度 / 指标。
在 P2RModel 中通过 hook 方式接入，不修改主训练逻辑。
"""
import json
import os
import torch
from datetime import datetime
from model.gates import GatedConv, GatedAttention


class DiagnoseLogger:
    """Collects training diagnostics and exports to JSON."""

    def __init__(self, log_dir="./diagnose_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.data = {
            "config": {},
            "model": {},
            "data": {},
            "per_epoch": [],
            "val_metrics": {},
        }
        self._epoch_data = None
        self._batch_counter = 0

    # ── Configuration ───────────────────────────────────────────────

    def log_config(self, model):
        """Record training config from model attributes."""
        fields = [
            'freeze_backbone', 'freeze_head', 'freeze_attention',
            'freeze_feature_fuse', 'gt_ratios_per_scene', 'pseudo',
            'reg_mode', 'beta', 'den_factor', 'lr', 'max_epochs',
            'training_mode', 'ST', 'dens_recon', 'use_attention_gate',
            'weight_decay', 'delta_L_mode', 'batch_size',
        ]
        self.data["config"] = {f: getattr(model, f, None) for f in fields}

    # ── Model stats ─────────────────────────────────────────────────

    def log_model_stats(self, model):
        """Record parameter counts and gate distribution."""
        student = model.student
        total = sum(p.numel() for p in student.parameters())
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        self.data["model"]["total_params"] = total
        self.data["model"]["trainable_params"] = trainable

        gate_count = 0
        gate_locations = {}
        for n, m in student.named_modules():
            if isinstance(m, GatedConv):
                gate_count += 1
                prefix = n.split('.')[0] if '.' in n else n
                gate_locations[prefix] = gate_locations.get(prefix, 0) + 1
            elif isinstance(m, GatedAttention):
                tag = f"{n}.gate_attn"
                gate_locations[tag] = 3  # q, k, v
        self.data["model"]["gate_count"] = gate_count
        self.data["model"]["gate_locations"] = gate_locations

    # ── Data info ───────────────────────────────────────────────────

    def log_data_info(self, model):
        """Record dataset and labeled_set info."""
        loader = model.train_loader
        if loader is None:
            return
        dataset = loader.dataset
        if hasattr(dataset, 'datasets'):
            self.data["data"]["train_samples"] = len(dataset)
        if model.labeled_set is not None:
            self.data["data"]["labeled_set_size"] = len(model.labeled_set._ids)
            self.data["data"]["labeled_set_counts"] = dict(model.labeled_set._counts)
            self.data["data"]["labeled_set_budget"] = dict(model.labeled_set._per_scene_max)

    # ── Gate value snapshots ────────────────────────────────────────

    def on_epoch_start(self, model):
        """Snapshot gate values at epoch start."""
        epoch = model.current_epoch
        self._epoch_data = {
            "epoch": epoch,
            "gate_stats": self._collect_gate_stats(model),
        }
        self._batch_counter = 0

    def _collect_gate_stats(self, model):
        """Collect per-module-group gate statistics."""
        groups = {}  # prefix → list of gate values

        for n, m in model.student.named_modules():
            prefix = n.split('.')[0] if '.' in n else n

            if isinstance(m, GatedConv):
                g = 2 * torch.sigmoid(m.gate)
                key = f"{prefix}(GatedConv)"
                if key not in groups:
                    groups[key] = []
                groups[key].extend(g.detach().cpu().tolist())

            elif isinstance(m, GatedAttention):
                for gate_name in ['q_gate_logit', 'k_gate_logit', 'v_gate_logit']:
                    g = 2 * torch.sigmoid(getattr(m, gate_name))
                    key = f"{prefix}.{gate_name}"
                    if key not in groups:
                        groups[key] = []
                    groups[key].extend(g.detach().cpu().tolist())

        stats = {}
        for group_name, vals in groups.items():
            if vals:
                t = torch.tensor(vals)
                stats[group_name] = {
                    "mean": round(t.mean().item(), 4),
                    "min": round(t.min().item(), 4),
                    "max": round(t.max().item(), 4),
                    "std": round(t.std().item(), 4),
                }
        return stats

    # ── Batch-level logging ─────────────────────────────────────────

    def on_batch_end(self, model, batch_idx):
        """Record loss components, grad norms, output sums (every N batches)."""
        self._batch_counter += 1
        # 每 50 batch 或第一个/最后一个 epoch 每 10 batch 抽样
        sample_interval = 10 if (self._epoch_data and self._epoch_data["epoch"] == 0) else 50
        if self._batch_counter % sample_interval != 0:
            return
        if self._epoch_data is None:
            return

        if "batches" not in self._epoch_data:
            self._epoch_data["batches"] = []

        batch_log = {"batch_idx": batch_idx}

        # Loss 分量
        logged = {}
        for k, v in model.trainer.logged_metrics.items():
            try:
                logged[k] = round(v.item() if hasattr(v, 'item') else float(v), 6)
            except (TypeError, ValueError, RuntimeError):
                logged[k] = str(v)
        batch_log["losses"] = logged

        # Gate 梯度范数
        grad_norms = {}
        for n, p in model.student.named_parameters():
            if ('.gate' in n or 'gate_logit' in n) and p.requires_grad:
                gn = p.grad.norm().item() if p.grad is not None else None
                if gn is not None:
                    grad_norms[n] = round(gn, 8)
        batch_log["grad_norms"] = grad_norms

        # 额外：打印最大的 N 个梯度值方便定位
        if grad_norms:
            sorted_gn = sorted(grad_norms.items(), key=lambda x: -x[1])
            batch_log["top5_grad_norms"] = sorted_gn[:5]

        # 密度图输出 sum（由 _supervised_loss 中存储）
        if hasattr(model, '_diag_pre_global_sum'):
            batch_log["pre_global_sum"] = round(model._diag_pre_global_sum, 4)
            batch_log["gt_global_sum"] = round(model._diag_gt_global_sum, 4)
            batch_log["decoder_raw_sum"] = round(model._diag_decoder_raw_sum, 4)
            batch_log["gt_scaled_sum"] = round(model._diag_gt_scaled_sum, 4)
            batch_log["decoder_vs_gt_ratio"] = round(
                model._diag_decoder_raw_sum / model._diag_gt_scaled_sum, 4
            ) if model._diag_gt_scaled_sum != 0 else None

        # 每帧 GT 人数 vs 密度图 sum（验证高斯归一化）
        if hasattr(model, '_diag_frame_people'):
            batch_log["frame_people"] = model._diag_frame_people
            batch_log["frame_gt_sums"] = [
                round(v, 4) for v in model._diag_frame_gt_sums
            ]
            # 每帧的 dmap_sum/人数 比值
            ratios = []
            for n_ppl, dsum in zip(model._diag_frame_people, model._diag_frame_gt_sums):
                if n_ppl > 0 and dsum > 0:
                    ratios.append(round(dsum / n_ppl, 4))
                else:
                    ratios.append(None)
            batch_log["frame_gt_ratio_ppl"] = ratios

        # 当前 batch 的 SID
        if hasattr(model, '_diag_sid'):
            batch_log["sid"] = model._diag_sid

        self._epoch_data["batches"].append(batch_log)

    # ── Epoch end ───────────────────────────────────────────────────

    def on_epoch_end(self, model):
        """Finalise epoch data."""
        if self._epoch_data is not None:
            # 记录 epoch 级 metrics
            logged = {}
            for k, v in model.trainer.logged_metrics.items():
                try:
                    logged[k] = round(v.item() if hasattr(v, 'item') else float(v), 6)
                except (TypeError, ValueError, RuntimeError):
                    logged[k] = str(v)
            self._epoch_data["logged_metrics"] = logged

            # 当前 learning rate
            try:
                lr = model.trainer.optimizers[0].param_groups[0]['lr']
                self._epoch_data["lr"] = round(lr, 8)
            except (IndexError, KeyError, AttributeError):
                pass

            # labeled_set 快照（epoch 末状态）
            if model.labeled_set is not None:
                self._epoch_data["labeled_set_size"] = len(model.labeled_set._ids)
                self._epoch_data["labeled_set_counts"] = dict(model.labeled_set._counts)
                # 记录前 3 个 SID 作为样本
                sid_sample = list(model.labeled_set._ids)[:3]
                self._epoch_data["labeled_set_sids_sample"] = sid_sample

            self.data["per_epoch"].append(self._epoch_data)
        self._epoch_data = None

    # ── Validation metrics ──────────────────────────────────────────

    def log_val_metrics(self, model):
        """Record validation metrics per epoch."""
        logged = {}
        for k, v in model.trainer.logged_metrics.items():
            if k.startswith('test/'):
                try:
                    logged[k] = round(v.item() if hasattr(v, 'item') else float(v), 6)
                except (TypeError, ValueError, RuntimeError):
                    logged[k] = str(v)
        if logged:
            epoch = model.current_epoch
            self.data["val_metrics"][f"epoch_{epoch}"] = logged

    # ── Report ──────────────────────────────────────────────────────

    def save(self, filename=None):
        """Write JSON report to disk."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnose_{ts}.json"
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        print(f"[Diagnose] Report saved to {path}")
        return path
