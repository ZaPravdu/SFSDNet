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
import torch.nn as nn

from misc.KPI_pool import Task_KPI_Pool
from model.gate_utils import add_gates_to_conv
from model.drnet.vic import Video_Individual_Counter
from model.hyper_model import HyperModel

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
        assert train_cfg.training_mode == 'supervised', (
            f'DRNetModel requires training_mode=supervised, '
            f'got {train_cfg.training_mode}')

        super().__init__(cfg, cfg_data, train_cfg, train_loader)

        # ── Core model ─────────────────────────────────────────────
        self.student = Video_Individual_Counter(cfg, cfg_data)
        self._load_pretrained_weights(self.weight_path)

        # ── KPI pool for adaptive loss weighting ───────────────────
        self.task_KPI = Task_KPI_Pool(
            task_setting={'den': ['gt_cnt', 'mae'],
                          'match': ['gt_pairs', 'pre_pairs']},
            maximum_sample=1000,
        )

        # ── Optional gate injection ────────────────────────────────
        if self.inject_gate:
            logger.info('Injecting gates into DRNet backbone')
            add_gates_to_conv(self.student.Extractor)

    # ── Weight loading ──────────────────────────────────────────────

    def _fix_checkpoint_keys(self, state_dict):
        """Fix ``Extractor.module.`` prefix from original DRNet pretrained weights."""
        return {k.replace('Extractor.module.', 'Extractor.'): v
                for k, v in state_dict.items()}

    # ── Loss dispatch ───────────────────────────────────────────────

    def calculate_loss(self, data, mode):
        # Support both collate_fn (2-tuple) and p2r_collate_fn (3-tuple).
        if len(data) == 3:
            images, _, targets = data
        elif len(data) == 2:
            images, targets = data
        else:
            raise ValueError(f'Unexpected batch length: {len(data)}')

        images = images.to(self.device)
        for t in targets:
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    t[k] = v.to(self.device, non_blocking=True)

        if mode == 'train':
            return self._train_loss(images, targets)
        else:
            return self._val_loss(images, targets)

    def _train_loss(self, images, targets):
        out = self.student(images, targets)
        pre_map, gt_den, correct_pairs, match_pairs, tp_cnt, _ = out
        counting_mse, matching_loss, hard_loss, _ = self.student.loss

        # KPI-weighted multi-task loss.
        pre_cnt = pre_map.sum()
        gt_cnt = gt_den.sum()
        self.task_KPI.add({
            'den': {'gt_cnt': gt_cnt,
                    'mae': max(0., float(gt_cnt - (gt_cnt - pre_cnt).abs()))},
            'match': {'gt_pairs': match_pairs, 'pre_pairs': correct_pairs},
        })
        kpi = self.task_KPI.query()

        losses = torch.stack([counting_mse, matching_loss + hard_loss])
        weight = torch.tensor([kpi['den'], kpi['match']]).to(losses.device)
        weight = -(1 - weight) * torch.log(weight + 1e-8)
        weight = (weight / weight.sum()).detach()
        total_loss = (weight * losses).sum()

        # Regularisation (if gates are injected).
        if self.inject_gate and self.reg_mode is not None:
            reg = self._compute_reg()
            total_loss = total_loss + float(self.beta) * reg
        else:
            reg = torch.tensor(0.)

        self.log_dict({
            'train_loss': total_loss, 'train/counting_mse': counting_mse,
            'train/matching_loss': matching_loss, 'train/hard_loss': hard_loss,
            'train/kpi_den': kpi['den'], 'train/kpi_match': kpi['match'],
            'train/reg': reg,
        }, on_step=True, on_epoch=True, sync_dist=True)

        # ── 记录诊断数据（供 diagnose log 读取）──
        self._diag_pre_global_sum = pre_map.detach().sum().item()
        self._diag_gt_global_sum = gt_den.detach().sum().item()
        self._diag_decoder_raw_sum = (pre_map.detach() * self.den_factor).sum().item()
        self._diag_gt_scaled_sum = (gt_den.detach() * self.den_factor).sum().item()
        self._diag_frame_people = [len(t['points']) for t in targets]
        self._diag_frame_gt_sums = [gt_den[i].detach().sum().item() for i in range(len(gt_den))]

        return total_loss

    def _val_loss(self, images, targets):
        pre_map, gt_den, matched = self.student.val_forward(images, targets)

        gt_cnt = gt_den.sum()
        pre_cnt = pre_map.sum()
        mae = (pre_cnt - gt_cnt).abs()
        mse = (pre_cnt - gt_cnt).pow(2)
        self.log_dict({
            'val/mae': mae, 'val/mse': mse,
            'val/pre_detail': pre_cnt,
            'val/gt_detail': gt_cnt,
        }, on_epoch=True, sync_dist=True)
        return {'val_loss': mae}

    # ── Gate regularisation ─────────────────────────────────────────

    def _compute_reg(self):
        total = torch.tensor(0., device=self.device)
        for m in self.student.modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, 'l2_regularization'):
                total = total + getattr(m, 'l2_regularization', lambda: 0.)()
        return total

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

        with pytest.raises(AssertionError, match='supervised'):
            DRNetModel(cfg, cfg_data, train_cfg)


class TestDRNetModelTrainingStep:
    """training_step 的分发逻辑和 loss 计算。"""

    BATCH_3 = None  # (weak, strong, targets) 格式
    BATCH_2 = None  # (images, targets) 格式

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

            # Student forward → 返回真实 Tensor，这样 pre_map.sum() 等操作正常工作
            m.student.return_value = (
                torch.tensor([[10.0]]),   # pre_map → sum=10
                torch.tensor([[20.0]]),   # gt_den → sum=20
                torch.tensor(5.0),         # correct_pairs
                torch.tensor(6.0),         # match_pairs
                torch.tensor(4.0),         # tp_cnt
                {},                        # matched_results
            )
            m.student.loss = (
                torch.tensor(1.0),  # counting_mse
                torch.tensor(0.5),  # matching_loss
                torch.tensor(0.2),  # hard_loss
                torch.tensor(0.0),  # norm_loss
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

    @pytest.fixture
    def batch_2(self):
        if TestDRNetModelTrainingStep.BATCH_2 is None:
            TestDRNetModelTrainingStep.BATCH_2 = (
                torch.randn(2, 3, 64, 64),
                [{'points': torch.zeros((0, 2))}, {'points': torch.zeros((0, 2))}],
            )
        return TestDRNetModelTrainingStep.BATCH_2

    def test_handles_3tuple_batch(self, model, batch_3):
        """3 元组 batch (p2r_collate_fn 输出) 应正确解包。"""
        loss = model.training_step(batch_3, 0)
        assert isinstance(loss, torch.Tensor)
        # student 收到的是 weak_imgs
        assert model.student.call_args[0][0] is batch_3[0]

    def test_handles_2tuple_batch(self, model, batch_2):
        """2 元组 batch (collate_fn 输出) 应正确解包。"""
        loss = model.training_step(batch_2, 0)
        assert isinstance(loss, torch.Tensor)
        assert model.student.call_args[0][0] is batch_2[0]

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
        with pytest.raises(ValueError, match='Unexpected batch length'):
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
        batch = (torch.randn(2, 3, 64, 64), [{}, {}])
        model.validation_step(batch, 0)
        model.student.val_forward.assert_called_once()

    def test_val_forward_no_frame_signal_error(self, model):
        """val_forward 不因 frame_signal 参数缺失而崩溃 (Bug #1 回归)。"""
        batch = (torch.randn(2, 3, 64, 64), [{}, {}])
        try:
            model.validation_step(batch, 0)
        except TypeError as e:
            if 'frame_signal' in str(e):
                pytest.fail(f'Bug #1: val_forward 仍要求 frame_signal 参数: {e}')
            raise


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
