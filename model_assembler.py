"""Factory and re-exports — create the right model per cfg.MODEL."""
from model.hyper_model import HyperModel
from model.p2r_model import P2RModel
from model.gates import BaseGatedModule, GatedConv, GatedAttention, GatedCrossAttention
from model.gate_utils import add_gates_to_conv, add_gates_to_attention, load_gate_freeze_config
from model.visualization import visualize_density_debug
from model.labeled_set import LabeledSet

__all__ = [
    'HyperModel', 'P2RModel',
    'BaseGatedModule', 'GatedConv', 'GatedAttention', 'GatedCrossAttention',
    'add_gates_to_conv', 'add_gates_to_attention',
    'load_gate_freeze_config', 'visualize_density_debug', 'LabeledSet',
]


def get_model(cfg, cfg_data, train_cfg, train_loader=None):
    """Factory: return a LightningModule for the configured model."""
    if cfg.MODEL == 'SDNet':
        return P2RModel(cfg, cfg_data, train_cfg, train_loader)
    if cfg.MODEL == 'DRNet':
        from model.drnet import DRNetModel
        return DRNetModel(cfg, cfg_data, train_cfg, train_loader)
    raise ValueError(f"Unknown cfg.MODEL: '{cfg.MODEL}'. Use 'SDNet' or 'DRNet'.")
