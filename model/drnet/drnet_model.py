"""DRNetModel — Lightning wrapper for DRNet Video Individual Counting.

Supports:
  * Supervised training with KPI-weighted multi-task loss.
  * Two-param-group optimiser (backbone LR + matching LR, ExponentialLR).
  * Optional gate injection for regularisation experiments.
  * Checkpoint serialisation (via HyperModel).
"""
import logging

import pytest
import torch
import torch.nn.functional as F

from misc.KPI_pool import Task_KPI_Pool
from model.gate_utils import add_gates_to_conv, delta_L
from model.drnet.vic import Video_Individual_Counter
from model.hyper_model import HyperModel
from model.fisher import compute_fisher

logger = logging.getLogger(__name__)


class DRNetModel(HyperModel):
    """Lightning wrapper for DRNet's Video_Individual_Counter.

    Parameters
    ----------
    cfg : edict
        Global config (must include ``MODEL``, ``NET``, ``FEATURE_DIM``,
        ``LR_Base``, ``LR_Thre``, ``LR_DECAY``, ``ROI_RADIUS``, etc.).
    cfg_data : edict
        Dataset config (``DEN_FACTOR``, ``TRAIN_SIZE``, etc.).
    train_cfg : SimpleNamespace | argparse.Namespace
        Training hyper-parameters:
        ``training_mode``, ``lr``, ``weight_decay``, ``max_epochs``,
        ``weight_path``, ``inject_gate``, ``reg_mode``, ``beta``.
    train_loader : DataLoader | None
        Needed when ``gt_ratios_per_scene > 0`` (not yet implemented).
    """
    def __init__(self, cfg, cfg_data, train_cfg, train_loader=None):
        assert cfg.MODEL == 'DRNet', f'DRNetModel requires MODEL=DRNet, got {cfg.MODEL}'
        assert train_cfg.training_mode in ('supervised', 'unsupervised', 'semi_supervised'), (
            f'DRNetModel requires training_mode in supervised/unsupervised/semi_supervised, '
            f'got {train_cfg.training_mode}')

        super().__init__(cfg, cfg_data, train_cfg, train_loader)

        # ── Core model ─────────────────────────────────────────────
        self.student = Video_Individual_Counter(cfg, cfg_data)

        # ── Teacher + freeze ───────────────────────────────────────
        self._setup_student_teacher()

        # ── Load pretrained weights (student + optional teacher) ───
        self._load_pretrained_weights(self.weight_path)
        self.downsample_factor = train_cfg.downsample_factor

        # ── KPI pool for adaptive loss weighting ───────────────────
        self.task_KPI = Task_KPI_Pool(
            task_setting={'den': ['gt_cnt', 'mae'],
                          'match': ['gt_pairs', 'pre_pairs']},
            maximum_sample=1000,
        )

        # ── Gate injection (after weight load) ─────────────────────
        self._inject_gates()

        # ── Frame prediction buffer (temporal consistency) ────────
        self.frame_buffer = {}
        self._val_count_errors = []
        self._val_labeled_errors = []
        self._val_unlabeled_errors = []

    # ── Weight loading ──────────────────────────────────────────────

    def _fix_checkpoint_keys(self, state_dict):
        """Fix ``Extractor.module.`` prefix from original DRNet pretrained weights."""
        return {k.replace('Extractor.module.', 'Extractor.'): v
                for k, v in state_dict.items()}

    # ── Init helpers ────────────────────────────────────────────────

    def _setup_student_teacher(self):
        """Create teacher if ST; freeze modules per config (matching SDNetModel pattern)."""
        if self.ST:
            self.teacher = Video_Individual_Counter(self.cfg, self.cfg_data)
            super()._setup_teacher(self.teacher)

        freeze_config = [
            (getattr(self, 'freeze_backbone', False), [
                self.student.Extractor.layer1,
                self.student.Extractor.layer2,
                self.student.Extractor.layer3,
                self.student.Extractor.neck,
                self.student.Extractor.neck2f,
            ]),
            (getattr(self, 'freeze_head', False), self.student.Extractor.loc_head),
            (getattr(self, 'freeze_feature_fuse', False), self.student.Extractor.feature_head),
        ]
        for flag, module in freeze_config:
            if flag:
                modules = module if isinstance(module, (list, tuple)) else [module]
                for m in modules:
                    for p in m.parameters():
                        p.requires_grad = False
        # freeze_attention is no-op on DRNet (no attention modules).

    def _inject_gates(self):
        """Inject GatedConv into Extractor; Gaussian layer is exempted (not part of Extractor)."""
        if not self.inject_gate:
            return
        if self.reg_mode in ('l1', 'l2'):
            logger.info('Injecting gates into DRNet Extractor')
            gate_mode = getattr(self, 'gate_mode', 'independent')
            for name in ['Extractor']:
                add_gates_to_conv(getattr(self.student, name), gate_mode=gate_mode)
        # DRNet has no Attention / CrossAttention — attention gate injection skipped.

    # ── Loss dispatch ───────────────────────────────────────────────

    def calculate_loss(self, data, mode):

        images, strong_img, targets = data

        images = images.to(self.device)
        strong_img = strong_img.to(self.device)
        for t in targets:
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    t[k] = v.to(self.device, non_blocking=True)

        if mode == 'val':
            return self._val_loss(images, targets)

        # Three-way dispatch matching P2RModel.
        if self.training_mode == 'supervised':
            return self._supervised_loss(images, targets)
        elif self.training_mode == 'unsupervised':
            return self._pseudo_loss(data)
        else:  # semi_supervised — per-batch gt_flag
            gt_flag = targets[0].get('gt_flag', True)
            if gt_flag:
                weak_loss = self._supervised_loss(images, targets, push_buffer=True)
                strong_loss = self._supervised_loss(strong_img, targets, push_buffer=False)
                return weak_loss + strong_loss
            else:
                return self._pseudo_loss(data)

    def _supervised_loss(self, images, targets, push_buffer=False):
        out = self.student(images, targets)
        pred_density, gt_density, match_loss, hard_loss, correct_pairs, match_pairs, tp_cnt, _ = out

        # Density MSE — both sides scaled by den_factor (SDNet convention)
        counting_mse = F.mse_loss(
            pred_density * self.den_factor, gt_density * self.den_factor)

        # KPI-weighted multi-task loss (match_loss + hard_loss = 1 unit).
        pre_cnt = pred_density.sum()
        gt_cnt = gt_density.sum()
        self.task_KPI.add({
            'den': {'gt_cnt': gt_cnt,
                    'mae': max(0., float(gt_cnt - (gt_cnt - pre_cnt).abs()))},
            'match': {'gt_pairs': match_pairs, 'pre_pairs': correct_pairs},
        })
        kpi = self.task_KPI.query()

        matching_total = match_loss + hard_loss
        losses = torch.stack([counting_mse, matching_total])
        weight = torch.tensor([kpi['den'], kpi['match']]).to(losses.device)
        weight = -(1 - weight) * torch.log(weight + 1e-8)
        weight = (weight / weight.sum()).detach()
        total_loss = (weight * losses).sum()

        # Regularisation (via HyperModel's _add_reg — supports L1/L2).
        total_loss = self._add_reg(total_loss)

        self.log_dict({
            'train_loss': total_loss, 'train/counting_mse': counting_mse,
            'train/matching_loss': match_loss, 'train/hard_loss': hard_loss,
            'train/kpi_den': kpi['den'], 'train/kpi_match': kpi['match'],
        }, on_step=True, on_epoch=True, sync_dist=True)

        # ── 记录诊断数据（供 diagnose log 读取）──
        self._diag_pre_global_sum = pred_density.detach().sum().item()
        self._diag_gt_global_sum = gt_density.detach().sum().item()
        self._diag_decoder_raw_sum = (pred_density.detach() * self.den_factor).sum().item()
        self._diag_gt_scaled_sum = (gt_density.detach() * self.den_factor).sum().item()
        self._diag_frame_people = [len(t['points']) for t in targets]
        self._diag_frame_gt_sums = [gt_density[i].detach().sum().item() for i in range(len(gt_density))]

        # ── Push to frame buffer (weak-aug predictions only) ─────────
        if push_buffer:
            for i, t in enumerate(targets):
                key = (t.get('scene_name'), t.get('frame'))
                self.frame_buffer[key] = pred_density[i].detach().cpu()

        return total_loss

    def _val_loss(self, images, targets):
        pre_map, gt_den, matched = self.student.val_forward(images, targets)

        gt_cnt = gt_den.sum()
        pre_cnt = pre_map.sum()
        mae = (pre_cnt - gt_cnt).abs()
        density_mse = F.mse_loss(pre_map, gt_den)

        log_dict = {
            'val/mae': mae,
            'val/density_mse': density_mse,
            'val/pre_cnt': pre_cnt,
        }

        gt_flag = targets[0].get('gt_flag', True)
        prefix = 'val_labeled' if gt_flag else 'val_unlabeled'
        log_dict[f'{prefix}/mae'] = mae
        log_dict[f'{prefix}/density_mse'] = density_mse
        log_dict[f'{prefix}/pre_cnt'] = pre_cnt

        self.log_dict(log_dict, on_epoch=True, sync_dist=True)

        # Accumulate for RMSE (per group).
        err = (pre_cnt - gt_cnt).detach()
        self._val_count_errors.append(err)
        if gt_flag:
            self._val_labeled_errors.append(err)
        else:
            self._val_unlabeled_errors.append(err)

        return {'val_loss': mae}

    def on_train_epoch_start(self):
        self.frame_buffer.clear()
        super().on_train_epoch_start()

    def on_validation_epoch_end(self):
        def _log_rmse(errors, key):
            if errors:
                sq = torch.stack(errors).pow(2)
                self.log(key, sq.mean().sqrt(), sync_dist=True)
                errors.clear()

        _log_rmse(self._val_count_errors, 'val/rmse')
        _log_rmse(self._val_labeled_errors, 'val_labeled/rmse')
        _log_rmse(self._val_unlabeled_errors, 'val_unlabeled/rmse')
        super().on_validation_epoch_end()

    # ── Delta-L dynamic regularisation ─────────────────────────────

    def _build_forward_fn(self):
        """构造 compute_fisher 用的 forward_fn 闭包。

        Contract:
            Type: orchestration (closure)
            Input: batch (weak_img, _, targets)
            Output: (pred_density, gt_density) — 归一化密度图, raw GT
            Skips: batches where gt_flag=False (unlabeled)
        """
        def forward_fn(batch):
            weak_img, _, targets = batch
            weak_img = weak_img.to(self.device)
            if self.training_mode != 'supervised' \
                    and not targets[0].get('gt_flag', True):
                logger.warning('[DRNetModel] forward_fn: skipping unlabeled batch (gt_flag=False)')
                return None
            out = self.student(weak_img, targets)
            return out[0], out[1].detach()
        return forward_fn

    def _compute_delta_L(self):
        """编排：Fisher estimation → reg_coeff 写入。

        Contract:
            Type: orchestration
            Calls: _build_forward_fn → _get_gate_params → compute_fisher → _apply_delta_L_coeffs
            Side effects: eval/train toggle, reg_coeff 修改
        """
        if not self.inject_gate:
            logger.warning('[DRNetModel] inject_gate=False, skipping delta-L')
            return
        if self.delta_L_mode is None:
            logger.warning('[DRNetModel] delta_L_mode=None, skipping delta-L')
            return
        self.student.eval()
        logger.info('[DRNetModel] Computing delta-L regularization')

        forward_fn = self._build_forward_fn()
        gate_params = self._get_gate_params()
        fisher, grad_mean = compute_fisher(
            forward_fn, self.train_loader, gate_params,
            self.device, mc_iters=3, quiet=False)

        self._apply_delta_L_coeffs(fisher, grad_mean)
        self.student.train()

    def _pseudo_loss(self, data):
        """Density-map pseudo-supervision loss.

        Teacher (ST=True, fast-fail if None) or student.eval() (ST=False)
        generates pseudo density maps. Student learns from strong-aug
        images via MSE loss with optional spatial sum-pool downsampling
        to cancel zero-mean noise.

        Parameters
        ----------
        data : tuple
            3-tuple (weak_img, strong_img, targets) or 2-tuple fallback.

        Returns
        -------
        torch.Tensor
            Scalar loss with gradient tracking.
        """
        # ── Unpack ──────────────────────────────────────────────────
  
        weak_img, strong_img, targets = data

        weak_img = weak_img.to(self.device)
        strong_img = strong_img.to(self.device)
        for t in targets:
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    t[k] = v.to(self.device, non_blocking=True)

        # ── Teacher: pseudo density map ─────────────────────────────
        with torch.no_grad():
            if self.ST:
                teacher_pre, *_ = self.teacher(weak_img, targets)
            else:
                self.student.eval()
                teacher_pre, *_ = self.student(weak_img, targets)
                self.student.train()

        # ── Buffer replacement ──────────────────────────────────────
        for i, t in enumerate(targets):
            key = (t.get('scene_name'), t.get('frame'))
            cached = self.frame_buffer.get(key)
            if cached is not None:
                teacher_pre[i] = cached.to(self.device)

        # ── Student: density prediction on strong aug ───────────────
        student_pre, *_ = self.student(strong_img, targets)

        # ── Density MSE (optional sum-pool downsampling) ────────────
        s = student_pre * self.den_factor
        t = teacher_pre * self.den_factor
        k = self.downsample_factor
        if k > 1:
            s = F.avg_pool2d(s, k) * (k ** 2)
            t = F.avg_pool2d(t, k) * (k ** 2)
        loss = F.mse_loss(s, t)

        # ── Gate regularisation (via HyperModel's _add_reg) ────────
        loss = self._add_reg(loss)

        # ── Diagnostics ────────────────────────────────────────────
        self._diag_pre_global_sum = student_pre.detach().sum().item()
        self._diag_gt_global_sum = teacher_pre.detach().sum().item()
        self._diag_decoder_raw_sum = (student_pre * self.den_factor).detach().sum().item()
        self._diag_gt_scaled_sum = (teacher_pre * self.den_factor).detach().sum().item()
        self._diag_frame_people = [len(t.get('points', [])) for t in targets]
        self._diag_frame_gt_sums = [teacher_pre[i].detach().sum().item() for i in range(len(teacher_pre))]

        self.log_dict({
            'train/pseudo_loss': loss,
            'train/pseudo_teacher_sum': teacher_pre.detach().sum(),
            'train/pseudo_student_sum': student_pre.detach().sum(),
        }, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    # ── Optimiser ──────────────────────────────────────────────────

    def configure_optimizers(self):
        backbone = [
            p for n, p in self.student.named_parameters()
            if 'Matching_Layer' not in n and p.requires_grad
        ]
        matching = list(self.student.Matching_Layer.parameters())

        assert backbone, 'No trainable backbone parameters found'

        opt = torch.optim.Adam([
            {'params': backbone, 'lr': float(self.lr),
             'weight_decay': float(self.weight_decay)},
            {'params': matching, 'lr': float(self.cfg.LR_Thre),
             'weight_decay': float(self.weight_decay)},
        ])
        sched = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=float(self.cfg.LR_DECAY))

        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched,
                                 'interval': 'epoch', 'frequency': 1}}


# ── Test helpers ────────────────────────────────────────────────────────────

def _make_train_cfg(**overrides):
    """Default training config for DRNet tests. Override via kwargs."""
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        training_mode='supervised', lr=5e-5, weight_decay=0.0, max_epochs=1,
        weight_path=None, inject_gate=False, reg_mode=None, beta=0.0,
        ST=False, downsample_factor=1,
        freeze_backbone=False, freeze_head=False, freeze_feature_fuse=False,
        freeze_attention=False, use_attention_gate=False, use_variance_reg=False,
        dens_recon=False, delta_L_mode=None, gate_freeze_json=None,
        gt_ratios_per_scene=0, batch_size=8,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── pytest-style tests ──────────────────────────────────────────────────────

class TestDRNetModelInit:
    """DRNetModel 构造时断言和配置校验。"""

    def test_requires_drnet_model(self):
        """MODEL != 'DRNet' 时必须抛异常。"""
        import pytest
        from unittest.mock import patch, Mock

        cfg = Mock()
        cfg.MODEL = 'SDNet'  # 错误值
        cfg_data = Mock()
        train_cfg = _make_train_cfg()

        with pytest.raises(AssertionError, match='DRNet'):
            DRNetModel(cfg, cfg_data, train_cfg)

    def test_requires_supervised_mode(self):
        """training_mode 不是 supervised 时必须抛异常。"""
        import pytest
        from unittest.mock import patch, Mock

        cfg = Mock()
        cfg.MODEL = 'DRNet'
        cfg_data = Mock()
        train_cfg = _make_train_cfg(training_mode='p2r')  # 错误值

        with pytest.raises(AssertionError, match='supervised/unsupervised/semi_supervised'):
            DRNetModel(cfg, cfg_data, train_cfg)


class TestDRNetModelTrainingStep:
    """training_step 的分发逻辑和 loss 计算。"""

    BATCH_3 = None  # (weak, strong, targets) 格式

    @pytest.fixture
    def model(self):
        from unittest.mock import patch, MagicMock

        with patch('model.drnet.drnet_model.Video_Individual_Counter'), \
             patch('model.drnet.drnet_model.Task_KPI_Pool'), \
             patch('model.drnet.drnet_model.add_gates_to_conv'):

            cfg = MagicMock()
            cfg.MODEL = 'DRNet'
            cfg.LR_Thre = 1e-2
            cfg.LR_DECAY = 0.95

            from types import SimpleNamespace
            cfg_data = SimpleNamespace(DEN_FACTOR=200)
            m = DRNetModel(cfg, cfg_data, _make_train_cfg())

            # Student forward → 8 元组（VIC 新格式）
            m.student.return_value = (
                torch.tensor([[10.0]]),   # pred_density → sum=10
                torch.tensor([[20.0]]),   # gt_density → sum=20
                torch.tensor(0.5),         # batch_match_loss
                torch.tensor(0.2),         # batch_hard_loss
                torch.tensor(5.0),         # correct_pairs
                torch.tensor(6.0),         # match_pairs
                torch.tensor(4.0),         # tp_cnt
                {},                        # matched_results
            )
            # KPI query → 返回真实 float，避免 torch.stack 对 MagicMock 报错
            m.task_KPI.query.return_value = {'den': 0.5, 'match': 0.5}
            return m

    @pytest.fixture
    def batch_3(self):
        if TestDRNetModelTrainingStep.BATCH_3 is None:
            TestDRNetModelTrainingStep.BATCH_3 = (
                torch.randn(2, 3, 64, 64),
                torch.randn(2, 3, 64, 64),
                [{'points': torch.zeros((0, 2))}, {'points': torch.zeros((0, 2))}],
            )
        return TestDRNetModelTrainingStep.BATCH_3


    def test_handles_3tuple_batch(self, model, batch_3):
        """3 元组 batch 应正确解包。"""
        loss = model.training_step(batch_3, 0)
        assert isinstance(loss, torch.Tensor)
        # student 收到的是 weak_imgs
        assert model.student.call_args[0][0] is batch_3[0]

    def test_loss_is_scalar(self, model, batch_3):
        """loss 必须是标量 Tensor。"""
        loss = model.training_step(batch_3, 0)
        assert loss.ndim == 0

    def test_kpi_pool_updated(self, model, batch_3):
        """KPI pool 应被更新（query 不应抛异常）。"""
        model.training_step(batch_3, 0)
        # KPI query 应能正常调用
        model.task_KPI.query.assert_called_once()

    def test_different_batch_length_raises(self, model):
        """意外的 batch 长度应抛 ValueError。"""
        import pytest
        bad_batch = (torch.randn(2, 3, 64, 64),)  # 只有 1 个元素
        with pytest.raises(ValueError):
            model.training_step(bad_batch, 0)


class TestDRNetModelValidationStep:
    """validation_step 的分发逻辑。"""

    @pytest.fixture
    def model(self):
        from unittest.mock import patch, MagicMock

        with patch('model.drnet.drnet_model.Video_Individual_Counter'), \
             patch('model.drnet.drnet_model.Task_KPI_Pool'):

            cfg = MagicMock()
            cfg.MODEL = 'DRNet'
            cfg.LR_Thre = 1e-2
            cfg.LR_DECAY = 0.95

            from types import SimpleNamespace
            cfg_data = SimpleNamespace(DEN_FACTOR=200)
            m = DRNetModel(cfg, cfg_data, _make_train_cfg())

            # Stub val_forward → 返回真实 Tensor
            m.student.val_forward.return_value = (
                torch.tensor([[10.0]]),  # pre_map
                torch.tensor([[20.0]]),  # gt_den
                {},                      # matched_results
            )
            return m

    def test_val_forward_called(self, model):
        """validation_step 应调用 student.val_forward (而非 forward)。"""
        batch = (torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64), [{}, {}])
        model.validation_step(batch, 0)
        model.student.val_forward.assert_called_once()

    def test_val_forward_no_frame_signal_error(self, model):
        """val_forward 不因 frame_signal 参数缺失而崩溃 (Bug #1 回归)。"""
        batch = (torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64), [{}, {}])
        try:
            model.validation_step(batch, 0)
        except TypeError as e:
            if 'frame_signal' in str(e):
                pytest.fail(f'Bug #1: val_forward 仍要求 frame_signal 参数: {e}')
            raise


class TestDRNetPseudoLoss:
    """_pseudo_loss 的契约和行为。"""

    B, C, H, W = 2, 1, 4, 4
    BATCH_3 = None

    @staticmethod
    def _teacher_out(val=0.0):
        """6-element tuple matching VIC.forward() output."""
        d = lambda: torch.full((TestDRNetPseudoLoss.B, 1, 4, 4), val, dtype=torch.float32)
        return d(), d(), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), {}

    @staticmethod
    def _student_out(val=0.0):
        """Same shape with grad tracking."""
        d = lambda: torch.full((TestDRNetPseudoLoss.B, 1, 4, 4), val, dtype=torch.float32).requires_grad_(True)
        return d(), d(), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), {}

    @pytest.fixture
    def model(self):
        from unittest.mock import patch, MagicMock

        with patch('model.drnet.drnet_model.Video_Individual_Counter'), \
             patch('model.drnet.drnet_model.Task_KPI_Pool'), \
             patch('model.drnet.drnet_model.add_gates_to_conv'):

            cfg = MagicMock()
            cfg.MODEL = 'DRNet'
            cfg.LR_Thre = 1e-2
            cfg.LR_DECAY = 0.95

            from types import SimpleNamespace
            cfg_data = SimpleNamespace(DEN_FACTOR=200)
            m = DRNetModel(cfg, cfg_data, _make_train_cfg(
                training_mode='unsupervised', downsample_factor=1))

            m.student.return_value = self._student_out(0.0)
            return m

    @pytest.fixture
    def batch_3(self):
        if TestDRNetPseudoLoss.BATCH_3 is None:
            TestDRNetPseudoLoss.BATCH_3 = (
                torch.randn(2, 3, self.H, self.W),
                torch.randn(2, 3, self.H, self.W),
                [{'points': torch.zeros((0, 2))}, {'points': torch.zeros((0, 2))}],
            )
        return TestDRNetPseudoLoss.BATCH_3

    def test_returns_scalar_tensor(self, model, batch_3):
        """返回标量 Tensor with grad."""
        loss = model._pseudo_loss(batch_3)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_student_path(self, model, batch_3):
        """ST=False: student.eval() for teacher, student.train() after."""
        model.ST = False
        model.teacher = None
        loss = model._pseudo_loss(batch_3)
        assert isinstance(loss, torch.Tensor)

    def test_teacher_path(self, model, batch_3):
        """ST=True + teacher: teacher called."""
        from unittest.mock import MagicMock
        model.ST = True
        model.teacher = MagicMock()
        model.teacher.return_value = self._teacher_out(0.0)
        model.student.return_value = self._student_out(1.0)
        loss = model._pseudo_loss(batch_3)
        model.teacher.assert_called_once()
        assert loss.item() > 0

    def test_st_teacher_none_errors(self, model, batch_3):
        """ST=True + teacher=None → AttributeError (fast-fail)."""
        model.ST = True
        model.teacher = None
        with pytest.raises((AttributeError, TypeError)):
            model._pseudo_loss(batch_3)

    def test_mse_nonzero_for_different_maps(self, model, batch_3):
        """teacher≠student → loss > 0."""
        from unittest.mock import MagicMock
        model.ST = True
        model.teacher = MagicMock()
        model.teacher.return_value = self._teacher_out(0.0)
        model.student.return_value = self._student_out(1.0)
        loss = model._pseudo_loss(batch_3)
        assert loss.item() > 0

    def test_downsample_factor_changes_loss(self, model, batch_3):
        """downsample_factor > 1 改变 loss 值。"""
        from unittest.mock import MagicMock
        model.ST = True
        model.teacher = MagicMock()
        model.teacher.return_value = self._teacher_out(0.5)
        model.student.return_value = self._student_out(0.3)

        model.downsample_factor = 1
        loss_pixel = model._pseudo_loss(batch_3).item()

        model.downsample_factor = 2
        loss_pool = model._pseudo_loss(batch_3).item()

        assert loss_pixel != loss_pool, (
            f'downsampled and pixel loss should differ '
            f'(pixel={loss_pixel}, pool={loss_pool})')

    def test_1tuple_raises(self, model):
        """len=1 抛 ValueError."""
        with pytest.raises(ValueError):
            model._pseudo_loss((torch.randn(2, 3, 4, 4),))


class TestDRNetModelOptimizer:
    """configure_optimizers 的参数组和调度器。"""

    @staticmethod
    def _make_model(**overrides):
        """创建 mock DRNetModel 并注入必要的参数。"""
        from unittest.mock import patch, MagicMock

        cfg = MagicMock()
        cfg.MODEL = 'DRNet'
        cfg.LR_Thre = 1e-2
        cfg.LR_DECAY = 0.95

        from types import SimpleNamespace
        cfg_data = SimpleNamespace(DEN_FACTOR=200)

        with patch('model.drnet.drnet_model.Video_Individual_Counter'), \
             patch('model.drnet.drnet_model.Task_KPI_Pool'):
            m = DRNetModel(cfg, cfg_data, _make_train_cfg(lr=5e-5, **overrides))
            # configure_optimizers 需要 student.named_parameters() 返回真实 Parameter
            m.student.named_parameters.return_value = [
                ('conv.weight', torch.nn.Parameter(torch.randn(3, 3))),
            ]
            m.student.Matching_Layer.parameters.return_value = [
                torch.nn.Parameter(torch.randn(1)),
            ]
            return m

    def test_two_param_groups(self):
        """优化器应有 2 个参数组（backbone + matching）。"""
        m = self._make_model()
        assert len(m.configure_optimizers()['optimizer'].param_groups) == 2

    def test_matching_group_uses_lr_thre(self):
        """匹配层的 LR 应使用 cfg.LR_Thre 而非自带的 lr。"""
        m = self._make_model()
        assert m.configure_optimizers()['optimizer'].param_groups[1]['lr'] == 1e-2

    def test_scheduler_is_exponential_lr(self):
        """调度器应为 ExponentialLR。"""
        m = self._make_model()
        from torch.optim.lr_scheduler import ExponentialLR
        assert isinstance(
            m.configure_optimizers()['lr_scheduler']['scheduler'], ExponentialLR)
