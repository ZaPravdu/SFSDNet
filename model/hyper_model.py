from pytorch_lightning import LightningModule
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


class HyperModel(LightningModule):
    """Base LightningModule with AdamW + CosineAnnealingLR."""
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode='val')

    def calculate_loss(self, data, mode):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.1),
            'interval': 'epoch', 'frequency': 1, 'name': 'cosine_annealing',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
