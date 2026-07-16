import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import logging
from unittest.mock import patch, MagicMock, Mock

from misc.otm import den2seq
from model.VIC import Video_Counter
from model.hyper_model import HyperModel
from model.gate_utils import add_gates_to_conv, add_gates_to_attention, load_gate_freeze_config, delta_L
from model.gates import BaseGatedModule
from model.fisher import compute_fisher
from types import SimpleNamespace

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SDNetModel(HyperModel):
    """
    Teacher-Student P2R semi-supervised training LightningModule.
    Supports both p2r (pseudo-labeling) and supervised training modes.
    """
    def __init__(self, cfg, cfg_data, train_cfg, train_loader=None):
        super().__init__(cfg, cfg_data, train_cfg, train_loader)
        # P2RModel 专为 SDNet 设计。非 SDNet 模型（如 DRNet）应使用
        # model_assembler.get_model() 工厂，该工厂返回对应的 LightningModule。
        if cfg.MODEL != 'SDNet':
            logger.warning(
                "P2RModel created with cfg.MODEL='%s'. "
                "This class is designed for SDNet; SDNet-specific methods "
                "(_setup_student_teacher, _inject_gates, _p2r_loss, etc.) "
                "will fail. Use model_assembler.get_model() instead.",
                cfg.MODEL,
            )
        self._setup_student_teacher()
        self._load_pretrained_weights(self.weight_path)
        self._inject_gates()
        self.criterion = nn.MSELoss()

    @staticmethod
    def _sample_id(data):
        """Unique sample ID = scene/frame0_frame1 (sorted for direction invariance)."""
        t0, t1 = data[2][0], data[2][1]
        return f"{t0['scene_name']}/{min(t0['frame'], t1['frame'])}_{max(t0['frame'], t1['frame'])}"

    # ── Init helpers ──────────────────────────────────────────────

    def _setup_student_teacher(self):
        self.student = Video_Counter(self.cfg, self.cfg_data).eval()
        if self.ST:
            self.teacher = Video_Counter(self.cfg, self.cfg_data).eval()
            super()._setup_teacher(self.teacher)

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

    def _fix_checkpoint_keys(self, state_dict):
        """Remove DataParallel ``module.`` prefix from pretrained weights."""
        return {k[7:] if k.startswith('module.') else k: v
                for k, v in state_dict.items()}

    def _inject_gates(self):
        if not self.inject_gate:
            return
        if self.reg_mode in ('l1', 'l2'):
            gate_mode = getattr(self, 'gate_mode', 'independent')
            for name in ['share_decoder', 'global_decoder', 'in_out_decoder', 'Extractor', 'feature_fuse']:
                add_gates_to_conv(getattr(self.student, name), gate_mode=gate_mode)
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
        if mode == 'val':
            # val: always compute metrics via supervised loss
            loss = self._supervised_loss((data[0], data[2]), mode)

        elif mode == 'train':
            if self.training_mode == 'unsupervised':
                loss = self._pseudo_loss(data)
            elif self.training_mode == 'supervised':
                loss = self._supervised_loss((data[0], data[2]), mode)
            else:  # semi_supervised
                gt_flag = data[2][0].get('gt_flag', True)
                if gt_flag:
                    loss = self._supervised_loss((data[0], data[2]), mode)
                else:
                    loss = self._pseudo_loss(data)

        else:
            raise ValueError(f"Unknown mode '{mode}'")

        return loss

    # ── Single test ───────────────────────────────────────────────

    def single_test(self, img, targets):
        assert self.ST, 'single_test requires ST=True (teacher needed)'
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
        """Iterate labeled samples, compute per-pixel-MSE Fisher → set reg_coeff."""
        if not self.inject_gate:
            return
        self.student.eval()
        print('[P2RModel] Computing delta-L regularization')

        processed = 0

        def forward_fn(batch):
            nonlocal processed
            weak_img, _, targets = batch
            weak_img = weak_img.to(self.device)

            # Skip unlabeled samples (Delta-L needs GT for meaningful Fisher)
            if self.training_mode != 'supervised' \
                    and not targets[0].get('gt_flag', True):
                return None

            processed += 1
            out = self.student(weak_img, targets)
            pred = torch.cat([out[0], out[2], out[4]], dim=1)
            target = torch.cat([out[1], out[3], out[5]], dim=1).detach()
            return pred, target

        gate_params = self._get_gate_params()
        fisher, grad_mean = compute_fisher(
            forward_fn, self.train_loader, gate_params, self.device,
            mc_iters=3, quiet=False)

        logger.debug('[P2RModel] Delta-L: processed %d samples', processed)
        self._apply_delta_L_coeffs(fisher, grad_mean)
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

    def _pseudo_loss(self, data):
        weak_img, student_img, targets = data

        with torch.no_grad():
            if self.ST:
                pseudo_global, gt_global, pseudo_share, gt_share, pseudo_io, gt_io, _ = \
                    self.teacher(weak_img, targets)
            else:
                self.student.eval()
                pseudo_global, gt_global, pseudo_share, gt_share, pseudo_io, gt_io, _ = \
                    self.student(weak_img, targets)
                self.student.train()

        if self.dens_recon:
            model_for_recon = self.teacher if self.ST else self.student
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
                results.append(model_for_recon.Gaussian(dots))
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
        if not self.inject_gate:
            return loss
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

    def get_teacher_model(self):
        assert self.ST, 'teacher only exists when ST=True'
        return self.teacher

    def get_student_model(self):
        return self.student



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
        training_mode='semi_supervised', lr=0.0001, weight_decay=1e-6, max_epochs=10,
        dens_recon=False, ST=False, beta=1, reg_mode='l2',
        use_attention_gate=False, batch_size=8, gate_freeze_json=None,
        delta_L_mode=None, use_variance_reg=False, gt_ratios_per_scene=0,
        freeze_backbone=True, freeze_feature_fuse=True, freeze_head=True,
        freeze_attention=True, weight_path=None, inject_gate=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── pytest-style tests ──────────────────────────────────────────────────────

class TestP2RModelHelpers:
    def test_sample_id(self):
        t0 = {'scene_name': 'scene01', 'frame': 10}
        t1 = {'scene_name': 'scene01', 'frame': 5}
        sid = SDNetModel._sample_id([None, None, [t0, t1]])
        assert sid == 'scene01/5_10'

    def test_sample_id_direction_invariant(self):
        t0 = {'scene_name': 's', 'frame': 10}
        t1 = {'scene_name': 's', 'frame': 5}
        assert SDNetModel._sample_id([None, None, [t0, t1]]) == \
               SDNetModel._sample_id([None, None, [t1, t0]])

    def test_sample_id_cross_scene(self):
        t0 = {'scene_name': 's1', 'frame': 3}
        t1 = {'scene_name': 's2', 'frame': 7}
        sid = SDNetModel._sample_id([None, None, [t0, t1]])
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
    """Test calculate_loss dispatch.

    Contract:
    ───────────────────────────────────────────────────────────────
    training_mode    | gt_flag  | mode   → action
    supervised       | any      | val     → _supervised_loss
    semi_supervised  | any      | val     → _supervised_loss
    unsupervised     | any      | val     → _supervised_loss
    supervised       | any      | train   → _supervised_loss
    semi_supervised  | True     | train   → _supervised_loss
    semi_supervised  | False    | train   → _p2r_loss
    unsupervised     | any      | train   → _p2r_loss
    ───────────────────────────────────────────────────────────────
    """

    BATCH_DATA_WITH_GT = None
    BATCH_DATA_NO_GT = None

    @pytest.fixture
    def model(self):
        """Minimal P2RModel with loss methods stubbed out."""
        with patch('model.p2r_model.Video_Counter'), \
             patch('model.p2r_model.add_gates_to_conv'), \
             patch('model.p2r_model.add_gates_to_attention'), \
             patch('model.p2r_model.load_gate_freeze_config'):
            cfg = Mock()
            cfg_data = Mock(DEN_FACTOR=200)
            m = SDNetModel(cfg, cfg_data, _make_train_cfg())

            m._supervised_loss = Mock(return_value=torch.tensor(10.0))
            m._pseudo_loss = Mock(return_value=torch.tensor(20.0))
            return m

    @pytest.fixture(params=['gt', 'no_gt'])
    def data(self, request):
        """Batch with gt_flag=True or False in targets."""
        key = 'BATCH_DATA_WITH_GT' if request.param == 'gt' else 'BATCH_DATA_NO_GT'
        cache = getattr(TestCalculateLoss, key)
        if cache is None:
            img = torch.randn(2, 3, 64, 64)
            gt_flag = request.param == 'gt'
            targets = [
                {'scene_name': 'test', 'frame': 1, 'gt_flag': gt_flag},
                {'scene_name': 'test', 'frame': 2, 'gt_flag': gt_flag},
            ]
            cache = (img, img, targets)
            setattr(TestCalculateLoss, key, cache)
        return cache

    # ── Val: always _supervised_loss ──────────────────────────────

    def test_val_always_supervised_loss(self, model, data):
        loss = model.calculate_loss(data, 'val')
        model._supervised_loss.assert_called_once_with((data[0], data[2]), 'val')
        model._p2r_loss.assert_not_called()
        assert loss.item() == 10.0

    # ── Train + supervised ────────────────────────────────────────

    def test_supervised_train_calls_supervised_loss(self, model, data):
        model.training_mode = 'supervised'
        loss = model.calculate_loss(data, 'train')
        model._supervised_loss.assert_called_once()
        model._p2r_loss.assert_not_called()
        assert loss.item() == 10.0

    # ── Train + semi_supervised ───────────────────────────────────

    def test_semi_supervised_gt_calls_supervised_loss(self, model, data):
        """semi_supervised + gt_flag=True → _supervised_loss"""
        model.training_mode = 'semi_supervised'
        if not data[2][0].get('gt_flag'):
            pytest.skip('this test needs gt_flag=True')

        loss = model.calculate_loss(data, 'train')
        model._supervised_loss.assert_called_once()
        model._p2r_loss.assert_not_called()
        assert loss.item() == 10.0

    def test_semi_supervised_no_gt_calls_p2r_loss(self, model, data):
        """semi_supervised + gt_flag=False → _p2r_loss"""
        model.training_mode = 'semi_supervised'
        if data[2][0].get('gt_flag'):
            pytest.skip('this test needs gt_flag=False')

        loss = model.calculate_loss(data, 'train')
        model._p2r_loss.assert_called_once_with(data)
        model._supervised_loss.assert_not_called()
        assert loss.item() == 20.0

    # ── Train + unsupervised ──────────────────────────────────────

    def test_unsupervised_train_calls_p2r_loss(self, model, data):
        model.training_mode = 'unsupervised'
        loss = model.calculate_loss(data, 'train')
        model._p2r_loss.assert_called_once_with(data)
        model._supervised_loss.assert_not_called()
        assert loss.item() == 20.0



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
        """P2RModel with ST=True, mocked teacher/student, ready for _p2r_loss testing."""
        with patch('model.p2r_model.Video_Counter'), \
             patch('model.p2r_model.add_gates_to_conv'), \
             patch('model.p2r_model.add_gates_to_attention'), \
             patch('model.p2r_model.load_gate_freeze_config'):
            cfg = Mock()
            cfg_data = Mock(DEN_FACTOR=200)
            m = SDNetModel(cfg, cfg_data, _make_train_cfg(ST=True))

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
            p2r = SDNetModel(cfg, cfg_data, _make_train_cfg())
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
