from pytorch_lightning import LightningModule
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


class HyperModel(LightningModule):
    """Base LightningModule with AdamW + CosineAnnealingLR.

    Teacher-student infrastructure (controlled by ``ST``):
      * ``_setup_teacher()`` — freeze & register teacher model
      * ``ema_update()`` — exponential moving average from student → teacher
      * ``on_train_batch_end()`` — calls ema_update when ``self.ST`` is set
      * Checkpoint save/load for teacher state.
    """

    def __init__(self):
        super().__init__()
        self.teacher = None
        self.ema_momentum = 0.998

    # ── Teacher-student helpers ────────────────────────────────────

    def _setup_teacher(self, teacher_model):
        """Register and freeze teacher model, only when ``self.ST`` is True."""
        if not self.ST:
            return
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad = False

    def ema_update(self):
        """Exponential moving average: teacher ← momentum * teacher + (1-momentum) * student."""
        if self.teacher is None:
            return
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
                pt.data.copy_(pt.data * self.ema_momentum + ps.data * (1.0 - self.ema_momentum))

    # ── Lightning hooks ────────────────────────────────────────────

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ST and self.teacher is not None:
            self.ema_update()

    def on_save_checkpoint(self, checkpoint):
        if self.teacher is not None:
            checkpoint['teacher'] = self.teacher.state_dict()
            checkpoint['ema_momentum'] = self.ema_momentum

    def on_load_checkpoint(self, checkpoint):
        if self.teacher is not None and 'teacher' in checkpoint:
            self.teacher.load_state_dict(checkpoint['teacher'])
            if 'ema_momentum' in checkpoint:
                self.ema_momentum = checkpoint['ema_momentum']

    # ── Standard training hooks ────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode='val')

    def calculate_loss(self, data, mode):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.1),
            'interval': 'epoch', 'frequency': 1, 'name': 'cosine_annealing',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
