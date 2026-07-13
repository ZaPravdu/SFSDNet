from pytorch_lightning import LightningModule
import torch
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from diagnose_logger import DiagnoseLogger

logger = logging.getLogger(__name__)


class HyperModel(LightningModule):
    """Base LightningModule for video crowd counting.

    Provides:
    * Unified init from train_cfg namespace
    * Teacher-student infrastructure (ST + EMA)
    * ``calculate_loss`` dispatch (training_step → calculate_loss)
    * Gate logging in ``on_train_start``
    * DiagnoseLogger hook chain
    * Student + teacher checkpoint serialisation
    * Weight loading with ``_fix_checkpoint_keys`` hook
    """

    def __init__(self, cfg, cfg_data, train_cfg, train_loader=None):
        super().__init__()
        # ── From train_cfg ──────────────────────────────────────────
        self.__dict__.update(vars(train_cfg))
        self.train_loader = train_loader
        self.cfg = cfg
        self.cfg_data = cfg_data
        self.den_factor = cfg_data.DEN_FACTOR
        self.labeled_set = None

        # ── Teacher-student (optional, gated by self.ST) ────────────
        self.teacher = None
        self.ema_momentum = 0.998

        # ── Gate regularisation state ────────────────────────────────
        self._reg_coeff_externally_set = False

        # ── Diagnose ────────────────────────────────────────────────
        self.diagnose = DiagnoseLogger()
        self.diagnose.log_config(self)

    # ── Loss dispatch ──────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.calculate_loss(batch, 'val')

    def calculate_loss(self, data, mode):
        """Override in subclass — return a scalar loss Tensor."""
        pass

    # ── Teacher-student helpers ────────────────────────────────────

    def _setup_teacher(self, teacher_model):
        """Register and freeze teacher model, only when ``self.ST`` is True."""
        if not self.ST:
            return
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad = False

    def ema_update(self):
        """Exponential moving average from student → teacher."""
        if self.teacher is None:
            return
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
                pt.data.copy_(pt.data * self.ema_momentum + ps.data * (1.0 - self.ema_momentum))

    # ── Weight loading ─────────────────────────────────────────────

    def _fix_checkpoint_keys(self, state_dict):
        """Override in subclass for key-name prefix differences."""
        return state_dict

    def _load_pretrained_weights(self, weight_path):
        """Load pretrained weights into student (and teacher if ``self.ST``)."""
        if weight_path is None:
            logger.warning('[HyperModel] No weight_path — training from scratch')
            return
        state = torch.load(weight_path, map_location='cpu')
        state = self._fix_checkpoint_keys(state)
        self.student.load_state_dict(state, strict=True)
        if self.ST:
            self.teacher.load_state_dict(state, strict=True)

    # ── Lightning hooks ────────────────────────────────────────────

    def on_fit_start(self):
        """延迟初始化：记录模型参数统计（依赖 self.student，必须在子类 __init__ 之后）。"""
        self.diagnose.log_model_stats(self)

    def on_train_start(self):
        """Log gate parameters and dataset info."""
        if self.inject_gate:
            gates = [(n, p) for n, p in self.student.named_parameters()
                     if 'gate' in n or 'gate_logit' in n]
            print(f'[HyperModel] Gate parameters ({len(gates)} total):')
            n_trainable = 0
            for name, p in gates:
                trainable = p.requires_grad
                print(f'  {name}: requires_grad={trainable}, shape={list(p.shape)}')
                if trainable:
                    n_trainable += 1
            print(f'  Trainable: {n_trainable}/{len(gates)}')
            assert n_trainable > 0, 'No trainable gate parameters — gates will never update!'
        self.diagnose.log_data_info(self)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ST and self.teacher is not None:
            self.ema_update()
        self.diagnose.on_batch_end(self, batch_idx)

    def on_train_epoch_start(self):
        if self.delta_L_mode is not None:
            self._compute_delta_L()
        elif not self._reg_coeff_externally_set:
            self._assert_default_reg_coeff()
        self.diagnose.on_epoch_start(self)

    def _compute_delta_L(self):
        """Override when delta_L_mode is active (e.g. P2RModel)."""
        pass

    def _assert_default_reg_coeff(self):
        """Override when BaseGatedModule reg_coeff needs verification (e.g. P2RModel)."""
        pass

    def on_train_epoch_end(self):
        self.diagnose.on_epoch_end(self)

    def on_validation_epoch_end(self):
        self.diagnose.log_val_metrics(self)

    # ── Checkpoint ─────────────────────────────────────────────────

    def on_save_checkpoint(self, checkpoint):
        if self.teacher is not None:
            checkpoint['teacher'] = self.teacher.state_dict()
            checkpoint['ema_momentum'] = self.ema_momentum
        checkpoint['student'] = self.student.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.teacher is not None and 'teacher' in checkpoint:
            self.teacher.load_state_dict(checkpoint['teacher'])
            if 'ema_momentum' in checkpoint:
                self.ema_momentum = checkpoint['ema_momentum']
        if 'student' in checkpoint:
            self.student.load_state_dict(checkpoint['student'])

    # ── Optimiser ──────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer, T_max=self.max_epochs,
                eta_min=self.lr * 0.1),
            'interval': 'epoch', 'frequency': 1, 'name': 'cosine_annealing',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
