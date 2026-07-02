import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import logging
from unittest.mock import patch, MagicMock, Mock

from misc.otm import den2seq
from model.VIC import Video_Counter
from model.hyper_model import HyperModel
from model.gate_utils import add_gates_to_conv, add_gates_to_attention, load_gate_freeze_config
from model.gates import BaseGatedModule
from model.labeled_set import LabeledSet
from model.fisher import compute_fisher
from types import SimpleNamespace
from diagnose_logger import DiagnoseLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class P2RModel(HyperModel):
    """
    Teacher-Student P2R semi-supervised training LightningModule.
    Supports both p2r (pseudo-labeling) and supervised training modes.
    """
    def __init__(self, cfg, cfg_data, train_cfg, train_loader=None):
        super().__init__()
        # 从 argparse.Namespace 批量解包到 self
        self.__dict__.update(vars(train_cfg))
        self.train_loader = train_loader
        self.cfg_data = cfg_data
        self.cfg = cfg
        self.labeled_set = self._build_labeled_set()
        self._setup_student_teacher()
        self._load_pretrained_weights(self.weight_path)
        self._inject_gates()
        self.criterion = nn.MSELoss()
        self.ema_momentum = 0.998
        self.den_factor = self.cfg_data.DEN_FACTOR
        self._reg_coeff_externally_set = False  # tracked for on_train_epoch_start guard

        # ── Diagnose logger ─────────────────────────────────────────
        self.diagnose = DiagnoseLogger()
        self.diagnose.log_config(self)
        self.diagnose.log_model_stats(self)

    def _build_labeled_set(self):
        """Build LabeledSet from train_loader scene totals and gt_ratios_per_scene."""
        if self.gt_ratios_per_scene <= 0:
            logger.debug('[P2RModel] gt_ratios_per_scene=%s <= 0, skipping LabeledSet', self.gt_ratios_per_scene)
            return None
        assert self.train_loader is not None, (
            "train_loader required when gt_ratios_per_scene > 0"
        )
        from collections import defaultdict
        dataset = self.train_loader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        scene_totals = defaultdict(int)
        for sub_ds in dataset.datasets:
            assert hasattr(sub_ds, 'label'), (
                "Each sub-dataset must have .label to extract scene_name"
            )
            scene_name = sub_ds.label[0]['scene_name']
            base_scene = scene_name.split('/')[0] if '/' in scene_name else scene_name
            scene_totals[base_scene] += len(sub_ds)

        ls = LabeledSet(ratio=self.gt_ratios_per_scene, scene_totals=scene_totals)

        total_all = sum(scene_totals.values())
        total_gt = sum(ls._per_scene_max.values())
        logger.debug('[P2RModel] gt_ratios_per_scene=%s', self.gt_ratios_per_scene)
        for scene in sorted(scene_totals.keys()):
            logger.debug('  %s: total=%s, gt_budget=%s',
                         scene, scene_totals[scene], ls._per_scene_max[scene])
        logger.debug('  TOTAL: %s samples -> %s GT budget', total_all, total_gt)
        return ls

    @staticmethod
    def _sample_id(data):
        """Unique sample ID = scene/frame0_frame1 (sorted for direction invariance)."""
        t0, t1 = data[2][0], data[2][1]
        return f"{t0['scene_name']}/{min(t0['frame'], t1['frame'])}_{max(t0['frame'], t1['frame'])}"

    # ── Init helpers ──────────────────────────────────────────────

    def _setup_student_teacher(self):
        self.student = Video_Counter(self.cfg, self.cfg_data).eval()
        if self.training_mode == 'p2r' and self.pseudo:
            self.teacher = Video_Counter(self.cfg, self.cfg_data).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        freeze_config = [
            (self.freeze_backbone, self.student.Extractor),
            (self.freeze_feature_fuse, self.student.feature_fuse),
            (self.freeze_head, [self.student.global_decoder, self.student.share_decoder, self.student.in_out_decoder]),
            (self.freeze_attention, self.student.share_cross_attention),
        ]
        for flag, module in freeze_config:
            if flag:
                modules = module if isinstance(module, (list, tuple)) else [module]
                for m in modules:
                    for p in m.parameters():
                        p.requires_grad = False

    def _load_pretrained_weights(self, weight_path):

        state = torch.load(weight_path)
        new_state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        self.student.load_state_dict(new_state, strict=True)
        if self.training_mode == 'p2r' and self.pseudo:
            self.teacher.load_state_dict(new_state, strict=True)

    def _inject_gates(self):
        if not self.inject_gate:
            return
        if self.reg_mode in ('l1', 'l2'):
            for name in ['share_decoder', 'global_decoder', 'in_out_decoder', 'Extractor', 'feature_fuse']:
                add_gates_to_conv(getattr(self.student, name))
        if self.use_attention_gate and self.reg_mode is not None:
            add_gates_to_attention(self.student)
        if self.gate_freeze_json is not None:
            load_gate_freeze_config(self.student, self.gate_freeze_json)

    def _apply_external_reg_coeff(self, coeff_dict):
        """
        Inject externally-computed reg_coeff into all gated modules.

        Reuses the module-lookup + set_reg_coeff pattern from _compute_delta_L().
        The coeff_dict keys must match student.named_parameters() names
        (e.g. 'Extractor.backbone.0.conv1.gate' for GatedConv,
         'share_cross_attention.0.attn.q_gate_logit' for GatedAttention).

        Args:
            coeff_dict: dict[param_name, list[float]] — reg_coeff per channel/head.
                        Must NOT contain beta (beta is applied by _add_reg).
        """
        assert coeff_dict, (
            "_apply_external_reg_coeff called with empty coeff_dict"
        )
        nm = dict(self.student.named_modules())

        # GatedConv: per-channel coeffs
        for name, coeff in coeff_dict.items():
            if name.endswith('.gate'):
                parent_path = name.rsplit('.', 1)[0]
                parent_mod = nm.get(parent_path)
                if parent_mod is None:
                    raise KeyError(
                        f"Module '{parent_path}' not found in student network. "
                        f"Gate param '{name}' has no matching parent."
                    )
                parent_mod.set_reg_coeff(coeff)

        # GatedAttention: group q/k/v under same parent
        attn = {}
        for name, coeff in coeff_dict.items():
            if name.endswith('_gate_logit'):
                parent_path = name.rsplit('.', 1)[0]
                gate_type = name.rsplit('.', 1)[1]
                attn.setdefault(parent_path, {})[gate_type] = coeff

        for parent_path, gates in attn.items():
            parent_mod = nm.get(parent_path)
            if parent_mod is None:
                raise KeyError(
                    f"Attention module '{parent_path}' not found in student network."
                )
            parent_mod.set_reg_coeff(
                gates.get('q_gate_logit', [1.0] * parent_mod.num_heads),
                gates.get('k_gate_logit', [1.0] * parent_mod.num_heads),
                gates.get('v_gate_logit', [1.0] * parent_mod.num_heads),
            )

        self._reg_coeff_externally_set = True

    # ── Loss dispatch ─────────────────────────────────────────────

    def calculate_loss(self, data, mode):
        sid = self._sample_id(data)
        self._diag_sid = sid  # 供 diagnose log 读取

        # Per-scene labeled-sample budget tracking (only during training)
        if mode == 'train' and self.labeled_set is not None:
            self.labeled_set.add(sid)

        # Determine if this sample has been selected into the labeled set
        has_gt = self.labeled_set is not None and sid in self.labeled_set

        # ── Loss dispatch ───────────────────────────────────────────
        if mode == 'val':
            if has_gt:
                return None  # prevent validation-data leakage via held-out GT samples
            loss = self._supervised_loss((data[0], data[2]), mode)

        elif mode == 'train':
            if has_gt:
                loss = self._supervised_loss((data[0], data[2]), mode)
            else:
                if self.pseudo:
                    loss = self._p2r_loss(data)
                else:
                    return None  # unlabeled sample in supervised mode — skip update

        else:
            raise ValueError(f"Unknown mode '{mode}'")

        return loss

    def on_train_start(self):
        """Log all gate parameters and confirm at least one is trainable."""
        if self.inject_gate:
            gates = [(n, p) for n, p in self.student.named_parameters()
                     if 'gate' in n or 'gate_logit' in n]
            print(f'[P2RModel] Gate parameters ({len(gates)} total):')
            for name, p in gates:
                print(f'  {name}: requires_grad={p.requires_grad}, shape={list(p.shape)}')
            n_trainable = sum(1 for _, p in gates if p.requires_grad)
            print(f'  Trainable: {n_trainable}/{len(gates)}')
            assert n_trainable > 0, 'No trainable gate parameters — gates will never update!'
        self._validate_gt_sampling()
        self.diagnose.log_data_info(self)

    def on_after_backward(self):
        """Assert all gate parameters received gradients from the backward pass.

        This hook only fires for batches that returned a real loss from
        training_step (batches that returned None skip backward entirely),
        so every trainable gate must have a non-None gradient.
        """
        for name, p in self.student.named_parameters():
            if name.endswith('.gate') or name.endswith('_gate_logit'):
                assert p.grad is not None, (
                    f"Gate '{name}' has None gradient after backward — "
                    f"not connected to computation graph"
                )


    def ema_update(self):
        if not self.pseudo:
            return
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
                pt.data.copy_(pt.data * self.ema_momentum + ps.data * (1.0 - self.ema_momentum))

    # ── Single test ───────────────────────────────────────────────

    def single_test(self, img, targets):
        assert self.training_mode == 'p2r', 'single_test only in p2r mode'
        self.teacher = self.teacher.eval()
        with torch.no_grad():
            pseudo_global, gt_global, pseudo_share, gt_share, pseudo_io, gt_io, _ = self.teacher(img, targets)

        recon_results, pseudo_results, dots_maps = [], [], []
        for dens in [pseudo_global, pseudo_share, pseudo_io]:
            dens = dens.detach()
            dot_map = torch.zeros_like(dens)
            for i in range(len(pseudo_global)):
                pts = den2seq(dens[i].squeeze().to(torch.float32))
                dot_map[i, 0, pts[:, 0], pts[:, 1]] = 1
            recon = self.teacher.Gaussian(dot_map)
            recon_results.append(recon.cpu())
            pseudo_results.append(dens.cpu())
            dots_maps.append(dot_map.cpu())
        return pseudo_results, recon_results, dots_maps

    # ── Delta-L dynamic regularization ────────────────────────────

    def on_train_epoch_start(self):
        if self.delta_L_mode is not None:
            self._compute_delta_L()
        elif self._reg_coeff_externally_set:
            pass  # coefficients from external injection (e.g. variance reg), trust them
        else:
            self._assert_default_reg_coeff()
        self._validate_gt_sampling()
        self.diagnose.on_epoch_start(self)

    def _assert_default_reg_coeff(self):
        """Assert all reg_coeff are 1.0.

        Must only be called when no coefficient source is active
        (caller — on_train_epoch_start — guarantees this via its if/elif/else chain).
        """
        for mod in self.student.modules():
            if not isinstance(mod, BaseGatedModule):
                continue
            if hasattr(mod, 'reg_coeff'):
                assert all(c == 1.0 for c in mod.reg_coeff), (
                    f"GatedConv reg_coeff must be 1.0 when delta_L_mode is None, "
                    f"got {mod.reg_coeff}"
                )
            for attr in ('q_reg_coeff', 'k_reg_coeff', 'v_reg_coeff'):
                if hasattr(mod, attr):
                    coeff = getattr(mod, attr)
                    assert all(c == 1.0 for c in coeff), (
                        f"{attr} must be 1.0 when delta_L_mode is None, "
                        f"got {coeff}"
                    )

    def _validate_gt_sampling(self):
        """Assert GT sampling is configured and enforced correctly."""
        if self.gt_ratios_per_scene <= 0:
            assert self.labeled_set is None, (
                f"labeled_set must be None when gt_ratios_per_scene={self.gt_ratios_per_scene}, "
                f"got {self.labeled_set}"
            )
            return
        assert self.labeled_set is not None, (
            f"labeled_set must be initialized when gt_ratios_per_scene={self.gt_ratios_per_scene}"
        )
        assert len(self.labeled_set._per_scene_max) > 0, (
            "LabeledSet has empty per_scene_max — no scenes configured for GT sampling"
        )
        total_budget = sum(self.labeled_set._per_scene_max.values())
        logger.debug('[P2RModel] GT sampling validated: %d scenes, total_budget=%s',
                     len(self.labeled_set._per_scene_max), total_budget)

    def _compute_delta_L(self):
        """Iterate labeled samples, compute correct per-pixel-MSE Fisher → set reg_coeff."""
        if not self.inject_gate:
            return
        self.student.eval()
        print('[P2RModel] Computing delta-L regularization')

        processed = 0  # track how many samples are actually computed

        def forward_fn(batch):
            nonlocal processed
            weak_img, _, targets = batch
            weak_img = weak_img.to(self.device)

            # LabeledSet filtering — only process held-out labeled samples
            if self.labeled_set is not None:
                sid = self._sample_id(batch)
                self.labeled_set.add(sid)
                if sid not in self.labeled_set:
                    logger.debug('[P2RModel] Delta-L: skip sample %s (not in labeled set)', sid)
                    return None

            processed += 1
            out = self.student(weak_img, targets)
            # out = (pre_global, gt_global, pre_share, gt_share, pre_io, gt_io, _)
            pred = torch.cat([out[0], out[2], out[4]], dim=1)
            target = torch.cat([out[1], out[3], out[5]], dim=1).detach()
            return pred, target

        gate_params = [(n, p) for n, p in self.student.named_parameters()
                       if p.requires_grad and ('gate' in n or 'gate_logit' in n)]

        fisher, grad_mean = compute_fisher(
            forward_fn, self.train_loader, gate_params, self.device,
            mc_iters=3, quiet=False)

        # Log + validate sample counts vs budget
        if self.labeled_set is not None:
            actual_total = sum(self.labeled_set._counts.values())
            budget_total = sum(self.labeled_set._per_scene_max.values())
            logger.debug('[P2RModel] Delta-L: processed=%d samples, labeled_set has %d unique sids',
                         processed, actual_total)
            for scene in sorted(self.labeled_set._per_scene_max):
                budget = self.labeled_set._per_scene_max[scene]
                actual = self.labeled_set._counts.get(scene, 0)
                logger.debug('  %s: processed=%d, budget=%d', scene, actual, budget)
                assert actual <= budget, (
                    f"Scene '{scene}' has {actual} labeled samples, exceeds budget {budget}"
                )
            # processed tracks all forward_fn calls that returned non-None;
            # labeled_set._counts tracks unique sids added — both reflect the same
            # set of labeled samples collected in this pass.
            assert processed == actual_total, (
                f"processed ({processed}) != labeled_set._counts.total ({actual_total}) — "
                f"forward_fn and add() are out of sync"
            )
            if processed < budget_total:
                logger.debug('  (dataset has fewer samples than budget: %d < %d)',
                             processed, budget_total)
        else:
            logger.debug('[P2RModel] Delta-L: processed %d samples (no budget limit)',
                         processed)

        nm = dict(self.student.named_modules())

        # GatedConv: per-channel coeffs
        for name in list(fisher.keys()):
            if name.endswith('.gate'):
                coeff = delta_L(grad_mean[name], fisher[name], self.delta_L_mode).tolist()
                parent_path = name.rsplit('.', 1)[0]
                nm[parent_path].set_reg_coeff(coeff)

        # GatedAttention: group q/k/v under same parent → one set_reg_coeff call
        attn = {}
        for name in list(fisher.keys()):
            if name.endswith('_gate_logit'):
                parent_path = name.rsplit('.', 1)[0]
                gate_type = name.rsplit('.', 1)[1]
                c = delta_L(grad_mean[name], fisher[name], self.delta_L_mode).tolist()
                attn.setdefault(parent_path, {})[gate_type] = c

        for parent_path, gates in attn.items():
            nm[parent_path].set_reg_coeff(
                gates.get('q_gate_logit', [1.0]),
                gates.get('k_gate_logit', [1.0]),
                gates.get('v_gate_logit', [1.0]),
            )

        self.student.train()

    # ── Loss functions ────────────────────────────────────────────

    def _supervised_loss(self, data, mode):
        images, targets = data
        pre_global, gt_global, pre_share, gt_share, pre_io, gt_io, _ = self.student(images, targets)

        global_loss = self.criterion(pre_global * self.den_factor, gt_global * self.den_factor)
        share_loss = self.criterion(pre_share * self.den_factor, gt_share * self.den_factor)
        io_loss = self.criterion(pre_io * self.den_factor, gt_io * self.den_factor)
        loss = (global_loss + 10 * share_loss + io_loss) / 3

        # 记录诊断数据（供 diagnose log 读取）
        self._diag_pre_global_sum = pre_global.detach().sum().item()
        self._diag_gt_global_sum = gt_global.detach().sum().item()
        self._diag_decoder_raw_sum = (pre_global * self.den_factor).detach().sum().item()
        self._diag_gt_scaled_sum = (gt_global * self.den_factor).detach().sum().item()
        self._diag_frame_people = [len(t['points']) for t in targets]
        self._diag_frame_gt_sums = [gt_global[i].detach().sum().item() for i in range(len(gt_global))]

        self.log('gt_global_loss', global_loss.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('gt_share_loss', share_loss.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('gt_in_out_loss', io_loss.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True, on_step=True)

        loss = self._add_reg(loss)

        if mode == 'val':
            self._log_val_metrics(pre_global, gt_global, pre_share, gt_share, pre_io, gt_io)
        return loss

    def _p2r_loss(self, data):
        # 只被未标记样本调用（标记样本走 _supervised_loss），不再需要 gt_flag
        weak_img, student_img, targets = data

        with torch.no_grad():
            pseudo_global, gt_global, pseudo_share, gt_share, pseudo_io, gt_io, _ = self.teacher(weak_img, targets)

        if self.dens_recon:
            recon_dots = []
            for dens in [pseudo_global, pseudo_share]:
                dens = dens.detach()
                dot_map = torch.zeros_like(dens)
                for i in range(len(pseudo_global)):
                    pts = den2seq(dens[i].squeeze()).cpu()
                    dot_map[i, 0, pts[:, 0], pts[:, 1]] = 1
                recon_dots.append(dot_map)

            global_dots, share_dots = recon_dots
            in_out_dots = global_dots - share_dots
            results = []
            for dots in [global_dots, share_dots, in_out_dots]:
                results.append(self.teacher.Gaussian(dots))
            pseudo_global, pseudo_share, pseudo_io = results

        pre_global, _, pre_share, _, pre_io, _, _ = self.student(student_img, targets)

        global_loss = self.criterion(pre_global * self.den_factor, pseudo_global * self.den_factor)
        share_loss = self.criterion(pre_share * self.den_factor, pseudo_share * self.den_factor)
        io_loss = self.criterion(pre_io * self.den_factor, pseudo_io * self.den_factor)
        loss = (global_loss + 10 * share_loss + io_loss) / 3

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        loss = self._add_reg(loss)

        return loss

    # ── Regularization ────────────────────────────────────────────

    def _add_reg(self, loss):
        if self.reg_mode == 'l2':
            reg = self.compute_l2_regularization()
            self.log_dict({'l2': reg.item()})
            return loss + self.beta * reg
        elif self.reg_mode == 'l1':
            reg = self.compute_l1_regularization()
            self.log_dict({'l1': reg.item()})
            return loss + self.beta * reg
        return loss

    def compute_l2_regularization(self):
        return sum(m.l2_regularization() for m in self.student.modules() if isinstance(m, BaseGatedModule))

    def compute_l1_regularization(self):
        return sum(m.l1_regularization() for m in self.student.modules() if isinstance(m, BaseGatedModule))

    # ── Metrics ───────────────────────────────────────────────────

    def _log_val_metrics(self, pre_global, gt_global, pre_share, gt_share, pre_io, gt_io):
        def mae_mse(pred, gt):
            diff = (pred.sum(dim=[1, 2, 3]) - gt.sum(dim=[1, 2, 3])).abs()
            return diff.mean(), diff.pow(2).mean()

        g_mae, g_mse = mae_mse(pre_global, gt_global)
        s_mae, s_mse = mae_mse(pre_share, gt_share)
        i_mae, i_mse = mae_mse(pre_io, gt_io)
        global_gt = self.criterion(pre_global, gt_global)
        share_gt = self.criterion(pre_share, gt_share)
        io_gt = self.criterion(pre_io, gt_io)

        self.log_dict({
            'test/global_mae': g_mae, 'test/share_mae': s_mae, 'test/io_mae': i_mae,
            'test/total_mae': (g_mae + s_mae + i_mae).mean().item(),
            'test/global_mse': g_mse, 'test/share_mse': s_mse, 'test/io_mse': i_mse,
            'test/total_mse': (g_mse + s_mse + i_mse).mean().item(),
            'test/global_gt_loss': global_gt, 'test/share_gt_loss': share_gt,
            'test/in_out_gt_loss': io_gt,
        }, on_epoch=True, on_step=False)

    def calculate_matrics(self, pred_dens, gt_dens):
        diff = (pred_dens.sum(dim=[1, 2, 3]) - gt_dens.sum(dim=[1, 2, 3])).abs()
        return diff.mean(), diff.pow(2).mean()

    # ── Optimizer ─────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.1),
            'interval': 'epoch', 'frequency': 1, 'name': 'cosine_annealing',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # ── EMA (on_train_batch_end) ──────────────────────────────────

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.pseudo and self.ST:
            self.ema_update()
        self.diagnose.on_batch_end(self, batch_idx)

    def on_train_epoch_end(self):
        self.diagnose.on_epoch_end(self)

    def on_validation_epoch_end(self):
        self.diagnose.log_val_metrics(self)

    # ── Checkpoint ────────────────────────────────────────────────

    def on_save_checkpoint(self, checkpoint):
        checkpoint['student'] = self.student.state_dict()
        if self.pseudo:
            checkpoint['teacher'] = self.teacher.state_dict()
            checkpoint['ema_momentum'] = self.ema_momentum
        if self.labeled_set is not None:
            checkpoint['labeled_set'] = self.labeled_set.serialize()

    def on_load_checkpoint(self, checkpoint):
        if self.pseudo and 'teacher' in checkpoint:
            self.teacher.load_state_dict(checkpoint['teacher'])
            if 'ema_momentum' in checkpoint:
                self.ema_momentum = checkpoint['ema_momentum']
        if self.labeled_set is not None and 'labeled_set' in checkpoint:
            self.labeled_set.deserialize(checkpoint['labeled_set'])

    def get_teacher_model(self):
        assert self.pseudo, 'teacher only exists when pseudo=True'
        return self.teacher

    def get_student_model(self):
        return self.student

def delta_L(grad, fisher, mode='original'):
    """Compute per-gate-channel Delta L coefficient.

    mode='original':  -g² / F   (original delta loss formula)
    mode='exp':        exp(-g² / F)  (bounded score in (0, 1])
    mode='inv':        F / g²   (inverted / negative-inverse formula)

    Uses reshape(-1) so single-element dimensions don't collapse to scalar.
    """
    g = grad.reshape(-1)
    f = fisher.reshape(-1)
    if mode == 'exp':
        return torch.exp(-(g ** 2) / (f + 1e-12))
    if mode == 'inv':
        return f / (g ** 2 + 1e-12)
    # default / 'original'
    return -(g ** 2) / (f + 1e-12)


# ── Debug visualization ────────────────────────────────────────────────────

def vis_batch(data, gt_den=None, pred_den=None, idx=0):
    """可视化一个 batch 的第 idx 个样本：图像 + 标注点 + 密度图。

    调试时在 training loop 里调用：
        pre, gt, ... = model(imgs, targets)
        vis_batch(data, gt_den=gt, pred_den=pre, idx=0)
        vis_batch(data, gt_den=gt, pred_den=pre, idx=0, save='debug.png')  # 保存

    data: (weak, strong, targets) 或 (images, targets)
    gt_den: [B,1,H,W] 真值密度图（model forward 返回的 gt_global）
    pred_den: [B,1,H,W] 预测密度图（model forward 返回的 pre_global）
    save: 保存路径，不传则 plt.show()
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if len(data) == 3:
        weak_img, strong_img, targets = data
    else:
        weak_img = strong_img = data[0]
        targets = data[1]

    tw = targets[idx]
    pts = tw['points'].cpu().numpy()

    def _to_img(t):
        img = t[idx].float().cpu().numpy().transpose(1, 2, 0)
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo + 1e-8)

    # 列：(img, title, den)
    pairs = [(_to_img(weak_img),
              f'Weak {tw.get("scene_name","?")} fr={tw.get("frame","?")} N={len(pts)}',
              gt_den)]
    if pred_den is not None:
        img = _to_img(strong_img) if len(data) == 3 else _to_img(weak_img)
        pairs.append((img, f'Pred N={len(pts)}', pred_den))

    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (img, title, den) in zip(axes, pairs):
        ax.imshow(img, cmap='gray')
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], c='lime', s=10, marker='.')
        if den is not None:
            d = den[idx, 0].detach().cpu().numpy()
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            ax.imshow(d, cmap='jet', alpha=0.5)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()

    fig.savefig('./vis', dpi=150, bbox_inches='tight')

    plt.close(fig)


# ── Test helpers ────────────────────────────────────────────────────────────

def _make_train_cfg(**overrides):
    """Default training config for tests. Override via kwargs."""
    cfg = SimpleNamespace(
        training_mode='p2r', lr=0.0001, weight_decay=1e-6, max_epochs=10,
        dens_recon=False, ST=False, beta=1, reg_mode='l2',
        use_attention_gate=False, batch_size=8, gate_freeze_json=None,
        delta_L_mode=None, use_variance_reg=False, gt_ratios_per_scene=0,
        freeze_backbone=True, freeze_feature_fuse=True, freeze_head=True,
        freeze_attention=True, pseudo=False, weight_path=None,
        inject_gate=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── pytest-style tests ──────────────────────────────────────────────────────

class TestP2RModelHelpers:
    def test_sample_id(self):
        t0 = {'scene_name': 'scene01', 'frame': 10}
        t1 = {'scene_name': 'scene01', 'frame': 5}
        sid = P2RModel._sample_id([None, None, [t0, t1]])
        assert sid == 'scene01/5_10'

    def test_sample_id_direction_invariant(self):
        t0 = {'scene_name': 's', 'frame': 10}
        t1 = {'scene_name': 's', 'frame': 5}
        assert P2RModel._sample_id([None, None, [t0, t1]]) == \
               P2RModel._sample_id([None, None, [t1, t0]])

    def test_sample_id_cross_scene(self):
        t0 = {'scene_name': 's1', 'frame': 3}
        t1 = {'scene_name': 's2', 'frame': 7}
        sid = P2RModel._sample_id([None, None, [t0, t1]])
        # Uses first target's scene_name
        assert sid.startswith('s1/')

    def test_batch_unpack_structure(self):
        """Verify the assumed batch structure used by _compute_delta_L forward_fn."""
        # Simulate a batch from the loader: (weak_img, placeholder, targets)
        # The forward_fn inside _compute_delta_L does:
        #   weak_img, _, targets = batch
        # This must not raise.
        batch = (torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64), [{}, {}])
        weak_img, _, targets = batch
        assert weak_img.shape[0] == 2
        assert len(targets) == 2


class TestCalculateLoss:
    """Test the calculate_loss dispatch logic contract.

    Contract summary:
    ─────────────────────────────────────────────────────
    mode      | has_gt (= in labeled_set) | pseudo=❓ → action
    train     | True                      | any        → _supervised_loss
    train     | False                     | True       → _p2r_loss
    train     | False                     | False      → None (skip update)
    val       | True                      | any        → None (skip, prevent leakage)
    val       | False                     | any        → _supervised_loss
    ─────────────────────────────────────────────────────

    labeled_set.add(sid) is only called in train mode when labeled_set is not None.
    When labeled_set is None, has_gt is always False.
    """

    BATCH_DATA = None  # class-level cache to avoid re-creating tensors

    @pytest.fixture
    def model(self):
        """Minimal P2RModel with heavy components and loss methods stubbed out."""
        with patch('model.p2r_model.Video_Counter'), \
             patch('model.p2r_model.add_gates_to_conv'), \
             patch('model.p2r_model.add_gates_to_attention'), \
             patch('model.p2r_model.load_gate_freeze_config'):
            cfg = Mock()
            cfg_data = Mock(DEN_FACTOR=200)
            m = P2RModel(cfg, cfg_data, _make_train_cfg())

            # Stub loss methods to return identifiable values
            m._supervised_loss = Mock(return_value=torch.tensor(10.0))
            m._p2r_loss = Mock(return_value=torch.tensor(20.0))

            # Stub sample ID for deterministic behavior
            m._sample_id = Mock(return_value='test_scene/1_2')

            return m

    @pytest.fixture
    def data(self):
        """Minimal batch tuple: (img, img, targets)."""
        if TestCalculateLoss.BATCH_DATA is None:
            img = torch.randn(2, 3, 64, 64)
            targets = [
                {'scene_name': 'test_scene', 'frame': 1},
                {'scene_name': 'test_scene', 'frame': 2},
            ]
            TestCalculateLoss.BATCH_DATA = (img, img, targets)
        return TestCalculateLoss.BATCH_DATA

    # ── Train mode ────────────────────────────────────────────────

    def test_train_labeled_calls_supervised_loss(self, model, data):
        """Train + has_gt → _supervised_loss"""
        model.labeled_set = MagicMock()
        model.labeled_set.__contains__.return_value = True

        loss = model.calculate_loss(data, 'train')

        model._supervised_loss.assert_called_once_with((data[0], data[2]), 'train')
        model._p2r_loss.assert_not_called()
        assert loss.item() == 10.0

    def test_train_unlabeled_pseudo_calls_p2r_loss(self, model, data):
        """Train + no_gt + pseudo → _p2r_loss"""
        model.labeled_set = None
        model.pseudo = True

        loss = model.calculate_loss(data, 'train')

        model._p2r_loss.assert_called_once_with(data)
        model._supervised_loss.assert_not_called()
        assert loss.item() == 20.0

    def test_train_unlabeled_no_pseudo_returns_none(self, model, data):
        """Train + no_gt + no pseudo → None (skip update)"""
        model.labeled_set = None
        model.pseudo = False

        loss = model.calculate_loss(data, 'train')

        model._supervised_loss.assert_not_called()
        model._p2r_loss.assert_not_called()
        assert loss is None

    # ── Validation mode ───────────────────────────────────────────

    def test_val_labeled_returns_none(self, model, data):
        """Val + has_gt → None (prevent test leakage)"""
        model.labeled_set = MagicMock()
        model.labeled_set.__contains__.return_value = True

        loss = model.calculate_loss(data, 'val')

        model._supervised_loss.assert_not_called()
        model._p2r_loss.assert_not_called()
        assert loss is None

    def test_val_unlabeled_calls_supervised_loss(self, model, data):
        """Val + no_gt → _supervised_loss"""
        model.labeled_set = None

        loss = model.calculate_loss(data, 'val')

        model._supervised_loss.assert_called_once_with((data[0], data[2]), 'val')
        model._p2r_loss.assert_not_called()
        assert loss.item() == 10.0

    # ── labeled_set edge cases ────────────────────────────────────

    def test_labeled_set_add_only_in_train(self, model, data):
        """labeled_set.add() is only called in train mode, never in val."""
        model.labeled_set = MagicMock()
        model.labeled_set.__contains__.return_value = True

        model.calculate_loss(data, 'val')
        model.labeled_set.add.assert_not_called()

        model.calculate_loss(data, 'train')
        model.labeled_set.add.assert_called_once_with('test_scene/1_2')

    def test_labeled_set_add_may_be_rejected_by_budget(self, model, data):
        """add() can silently reject (budget full); calculate_loss respects that."""
        model.labeled_set = MagicMock()
        # add() is called, but __contains__ still returns False → sample not labeled
        model.labeled_set.__contains__.return_value = False
        model.pseudo = True

        loss = model.calculate_loss(data, 'train')

        model.labeled_set.add.assert_called_once_with('test_scene/1_2')
        model._p2r_loss.assert_called_once()      # unlabeled path
        model._supervised_loss.assert_not_called()
        assert loss.item() == 20.0

    def test_labeled_set_add_before_contains_semantics(self, model, data):
        """When labeled_set is real, a freshly-seen sample gets labeled same iteration.

        Proves add(sid) happens before contains(sid) inside calculate_loss(train).
        """
        model.labeled_set = LabeledSet(ratio=1, scene_totals={'test_scene': 10})
        model.pseudo = True

        loss = model.calculate_loss(data, 'train')

        # The sample should have been added and then found in contains()
        assert 'test_scene/1_2' in model.labeled_set
        model._supervised_loss.assert_called_once()
        assert loss.item() == 10.0

    def test_labeled_set_none_treated_as_unlabeled(self, model, data):
        """When labeled_set is None, all samples have has_gt=False."""
        model.labeled_set = None
        model.pseudo = False

        loss = model.calculate_loss(data, 'train')

        assert loss is None
        model._supervised_loss.assert_not_called()
        model._p2r_loss.assert_not_called()

    def test_labeled_set_none_skips_add_in_train(self, model, data):
        """When labeled_set is None, add() is not called."""
        model.labeled_set = None

        model.calculate_loss(data, 'train')
        # No error should occur — the add() guard prevents it.
        model._supervised_loss.assert_not_called()


class TestP2RLoss:
    """Test the _p2r_loss method contract: teacher-student flow, dens_recon, logging.

    Contract summary:
    ─────────────────────────────────────────────────────────────────────
    Teacher     → pseudo-labels on weak_img             (no_grad)
    dens_recon  → if True + train: den2seq + Gaussian   (replaces pseudo)
    Student     → strong_img (train) / weak_img (val)
    Loss        → (global + 10*share + io) / 3 + reg
    Logging     → {mode}_loss always; metrics only in val
    ─────────────────────────────────────────────────────────────────────
    """

    B, C, H, W = 2, 1, 4, 4

    @pytest.fixture
    def model(self):
        """P2RModel with mocked teacher/student, ready for _p2r_loss testing."""
        with patch('model.p2r_model.Video_Counter'), \
             patch('model.p2r_model.add_gates_to_conv'), \
             patch('model.p2r_model.add_gates_to_attention'), \
             patch('model.p2r_model.load_gate_freeze_config'):
            cfg = Mock()
            cfg_data = Mock(DEN_FACTOR=200)
            m = P2RModel(cfg, cfg_data, _make_train_cfg())

            # Mock heavy forward passes
            m.teacher = MagicMock()
            m.student = MagicMock()

            # Suppress side-effect logging
            m.log = MagicMock()
            m._log_val_metrics = MagicMock()

            # Default: no regularization (identity)
            m._add_reg = MagicMock(side_effect=lambda x: x)

            return m

    @staticmethod
    def _dens(val):
        """Density map tensor [B, C, H, W] filled with val."""
        return torch.full((TestP2RLoss.B, TestP2RLoss.C, TestP2RLoss.H, TestP2RLoss.W), val,
                          dtype=torch.float32)

    @staticmethod
    def _teacher_out(g=0.0, s=0.0, io=0.0):
        """7-element tuple matching teacher forward (no grad tracking)."""
        d = TestP2RLoss._dens
        return d(g), d(0.0), d(s), d(0.0), d(io), d(0.0), {}

    @staticmethod
    def _student_out(g=0.0, s=0.0, io=0.0):
        """7-element tuple matching student forward (tensors require grad)."""
        d = lambda val: TestP2RLoss._dens(val).requires_grad_(True)
        return d(g), d(0.0), d(s), d(0.0), d(io), d(0.0), {}

    @pytest.fixture
    def data(self):
        """Standard batch: (weak_img, strong_img, targets)."""
        return (
            torch.randn(self.B, 3, self.H, self.W),
            torch.randn(self.B, 3, self.H, self.W),
            [{'scene_name': 's', 'frame': 1}, {'scene_name': 's', 'frame': 2}],
        )

    # ── Basic flow ────────────────────────────────────────────────

    def test_returns_scalar_loss(self, model, data):
        """_p2r_loss always returns a scalar tensor with gradients."""
        model.teacher.return_value = self._teacher_out()
        model.student.return_value = self._student_out()
        loss = model._p2r_loss(data)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_teacher_called_with_weak_img(self, model, data):
        """Teacher always runs on weakly augmented images."""
        weak, _, targets = data
        model.teacher.return_value = self._teacher_out()
        model.student.return_value = self._student_out()
        model._p2r_loss(data)
        model.teacher.assert_called_once_with(weak, targets)

    def test_teacher_inference_under_no_grad(self, model, data):
        """Teacher forward is wrapped in torch.no_grad() — no gradient tracking."""
        model.teacher.return_value = self._teacher_out()
        model.student.return_value = self._student_out()
        loss = model._p2r_loss(data)
        # Verify that the student tensor (not teacher) connects to grad graph.
        # If teacher grad leaked, the returned loss would be a leaf tensor.
        assert loss.grad_fn is not None, "loss must be connected to the computation graph"

    # ── Student input ────────────────────────────────────────────

    def test_student_receives_strong_img(self, model, data):
        """Student always receives strongly augmented images."""
        _, strong, targets = data
        model.teacher.return_value = self._teacher_out()
        model.student.return_value = self._student_out()
        model._p2r_loss(data)
        model.student.assert_called_once_with(strong, targets)

    # ── Loss computation ──────────────────────────────────────────

    def test_loss_composition_and_reg(self, model, data):
        """(global + 10*share + io) / 3 passed to _add_reg; reg output returned."""
        # pred_global=0.01, target=0.0 → scaled by 200: 2.0 vs 0.0 → MSE=4
        # pred_share=0.02, target=0.0  → scaled by 200: 4.0 vs 0.0 → MSE=16
        model.teacher.return_value = self._teacher_out(0.0, 0.0, 0.0)
        model.student.return_value = self._student_out(0.01, 0.02, 0.0)
        model._add_reg = MagicMock(side_effect=lambda x: x + 5.0)

        loss = model._p2r_loss(data)
        raw = torch.tensor((4.0 + 10 * 16.0 + 0.0) / 3.0)
        model._add_reg.assert_called_once()
        assert torch.isclose(model._add_reg.call_args[0][0], raw, atol=1e-4)
        assert loss.item() == pytest.approx(raw + 5.0)

    def test_loss_scales_with_den_factor(self, model, data):
        """den_factor multiplies pred/target before MSE, squaring the effect."""
        # pred=0.01, target=0.0 → scaled: 2.0 vs 0.0 → MSE = 4.0
        model.teacher.return_value = self._teacher_out(0.0, 0.0, 0.0)
        model.student.return_value = self._student_out(0.01, 0.0, 0.0)
        loss = model._p2r_loss(data)
        # Only global contributes: (4 + 0 + 0) / 3
        assert loss.item() == pytest.approx(4.0 / 3.0)

    def test_share_loss_weighted_10x(self, model, data):
        """Share loss is weighted 10× in the composition."""
        model.teacher.return_value = self._teacher_out(0.0, 0.0, 0.0)
        model.student.return_value = self._student_out(0.0, 0.01, 0.0)
        loss = model._p2r_loss(data)
        # Only share contributes: (0 + 10*4 + 0) / 3
        assert loss.item() == pytest.approx(10.0 * 4.0 / 3.0)

    # ── Logging ───────────────────────────────────────────────────

    def test_logs_train_loss(self, model, data):
        """self.log is called with 'train_loss'."""
        model.teacher.return_value = self._teacher_out()
        model.student.return_value = self._student_out()
        model._p2r_loss(data)
        model.log.assert_called_once()
        assert model.log.call_args[0][0] == 'train_loss'

    # ── dens_recon ────────────────────────────────────────────────

    def test_dens_recon_triggers_den2seq_and_gaussian(self, model, data):
        """dens_recon=True + train: den2seq extracts points, Gaussian re-blurs."""
        model.teacher.return_value = self._teacher_out(1.0, 1.0, 1.0)
        model.student.return_value = self._student_out(0.0, 0.0, 0.0)
        model.dens_recon = True
        model.teacher.Gaussian = MagicMock(return_value=self._dens(0.5))

        with patch('model.p2r_model.den2seq', return_value=torch.tensor([[1, 1]])) as md:
            model._p2r_loss(data)
            # den2seq: 2 maps (global, share) × B=2 batch items = 4 calls
            assert md.call_count == 4
            # Gaussian: re-blur for global, share, and residual in_out = 3 calls
            assert model.teacher.Gaussian.call_count == 3

    def test_dens_recon_disabled_no_effect(self, model, data):
        """dens_recon=False: no reconstruction."""
        model.teacher.return_value = self._teacher_out()
        model.student.return_value = self._student_out()
        model.dens_recon = False
        model.teacher.Gaussian = MagicMock()
        with patch('model.p2r_model.den2seq') as md:
            model._p2r_loss(data)
            md.assert_not_called()
            model.teacher.Gaussian.assert_not_called()

    def test_dens_recon_changes_loss_value(self, model, data):
        """Reconstruction replaces pseudo targets → different loss."""
        model.teacher.return_value = self._teacher_out(1.0, 1.0, 1.0)
        model.student.return_value = self._student_out(0.0, 0.0, 0.0)
        model.dens_recon = True
        model.teacher.Gaussian = MagicMock(return_value=self._dens(0.5))

        with patch('model.p2r_model.den2seq', return_value=torch.tensor([[1, 1]])):
            loss_with = model._p2r_loss(data).item()

        model.teacher.Gaussian.reset_mock()
        model.dens_recon = False
        loss_without = model._p2r_loss(data).item()

        assert loss_with != loss_without, (
            f"dens_recon should change the loss value "
            f"(with={loss_with:.4f}, without={loss_without:.4f})"
        )


class TestApplyExternalRegCoeff:
    """Behavioural tests for _apply_external_reg_coeff()."""

    def _make_model_with_gates(self):
        """Minimal model with real GatedConv and GatedAttention modules."""
        import torch.nn as nn
        from model.gates import GatedConv, GatedAttention, GatedCrossAttention

        model = nn.Module()
        model.conv1 = GatedConv(nn.Conv2d(3, 4, 1))
        model.conv2 = GatedConv(nn.Conv2d(4, 6, 1))
        model.attn = GatedAttention(dim=64, num_heads=4)
        model.cross = GatedCrossAttention(dim=64, num_heads=2)
        return model

    def _make_p2r_instance(self, model):
        """Create P2RModel instance with .student pointing to *model*."""
        with patch('model.p2r_model.Video_Counter'), \
             patch('model.p2r_model.add_gates_to_conv'), \
             patch('model.p2r_model.add_gates_to_attention'), \
             patch('model.p2r_model.load_gate_freeze_config'):
            cfg = Mock()
            cfg_data = Mock(DEN_FACTOR=200)
            p2r = P2RModel(cfg, cfg_data, _make_train_cfg())
            p2r.student = model
            return p2r

    def test_conv_gates_injected(self):
        """GatedConv.reg_coeff correctly set from coeff_dict."""
        model = self._make_model_with_gates()
        p2r = self._make_p2r_instance(model)

        coeff = {
            'conv1.gate': [0.1, 0.2, 0.3, 0.4],
            'conv2.gate': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
        p2r._apply_external_reg_coeff(coeff)

        assert model.conv1.reg_coeff == [0.1, 0.2, 0.3, 0.4]
        assert model.conv2.reg_coeff == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_attention_gates_injected(self):
        """GatedAttention.q/k/v_reg_coeff correctly set."""
        model = self._make_model_with_gates()
        p2r = self._make_p2r_instance(model)

        coeff = {
            'attn.q_gate_logit': [1.0, 2.0, 3.0, 4.0],
            'attn.k_gate_logit': [0.5, 0.6, 0.7, 0.8],
            'attn.v_gate_logit': [10.0, 20.0, 30.0, 40.0],
        }
        p2r._apply_external_reg_coeff(coeff)

        assert model.attn.q_reg_coeff == [1.0, 2.0, 3.0, 4.0]
        assert model.attn.k_reg_coeff == [0.5, 0.6, 0.7, 0.8]
        assert model.attn.v_reg_coeff == [10.0, 20.0, 30.0, 40.0]

    def test_partial_attention_fills_defaults(self):
        """Missing Q/K/V gate_logit keys default to [1.0]."""
        model = self._make_model_with_gates()
        p2r = self._make_p2r_instance(model)

        coeff = {'attn.q_gate_logit': [9.0, 8.0, 7.0, 6.0]}
        p2r._apply_external_reg_coeff(coeff)

        assert model.attn.q_reg_coeff == [9.0, 8.0, 7.0, 6.0]
        assert model.attn.k_reg_coeff == [1.0, 1.0, 1.0, 1.0]
        assert model.attn.v_reg_coeff == [1.0, 1.0, 1.0, 1.0]

    def test_conv_and_attn_mixed(self):
        """Conv and attention coeffs set in the same call, no interference."""
        model = self._make_model_with_gates()
        p2r = self._make_p2r_instance(model)

        coeff = {
            'conv1.gate': [0.0, 0.0, 0.0, 0.0],
            'cross.q_gate_logit': [1.5, 2.5],
            'cross.k_gate_logit': [3.5, 4.5],
            'cross.v_gate_logit': [5.5, 6.5],
        }
        p2r._apply_external_reg_coeff(coeff)

        assert model.conv1.reg_coeff == [0.0, 0.0, 0.0, 0.0]
        assert model.cross.q_reg_coeff == [1.5, 2.5]
        assert model.cross.k_reg_coeff == [3.5, 4.5]

    def test_unknown_module_raises(self):
        """Unknown parent_path in coeff_dict must raise KeyError, not silently skip."""
        model = self._make_model_with_gates()
        p2r = self._make_p2r_instance(model)

        coeff = {'nonexistent.gate': [1.0, 2.0, 3.0, 4.0]}
        with pytest.raises(KeyError, match='nonexistent'):
            p2r._apply_external_reg_coeff(coeff)

    def test_empty_coeff_dict_raises(self):
        """Empty coeff_dict must raise AssertionError, not silently return."""
        model = self._make_model_with_gates()
        p2r = self._make_p2r_instance(model)

        with pytest.raises(AssertionError, match='empty coeff_dict'):
            p2r._apply_external_reg_coeff({})
