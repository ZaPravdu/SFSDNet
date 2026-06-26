import torch.nn as nn


class BaseGatedModule(nn.Module):
    """Abstract base for all gated modules (GatedConv, GatedAttention, GatedCrossAttention)."""
    def l2_regularization(self):
        raise NotImplementedError

    def l1_regularization(self):
        raise NotImplementedError

    def apply_freeze_mask(self):
        raise NotImplementedError
