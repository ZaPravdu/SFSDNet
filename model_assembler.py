"""Thin re-exporter — all logic moved to model/ submodules."""
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
