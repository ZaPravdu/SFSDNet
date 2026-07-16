import json
import re

import torch
import torch.nn as nn

from model.ViT.models_crossvit import Attention, CrossAttention
from model.gates import GatedConv, GatedAttention, GatedCrossAttention, BaseGatedModule


def add_gates_to_conv(model, mode=None, gate_mode='independent'):
    """Replace all nn.Conv2d with GatedConv. ``mode`` accepted for backward compat."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(model, name, GatedConv(child, gate_mode=gate_mode))
        elif len(list(child.children())):
            add_gates_to_conv(child, mode=mode, gate_mode=gate_mode)


def add_gates_to_attention(model):
    """Replace Attention→GatedAttention, CrossAttention→GatedCrossAttention.
    Copies pretrained weights and requires_grad from originals."""
    for name, child in list(model.named_children()):
        if isinstance(child, Attention) and not isinstance(child, BaseGatedModule):
            gated = GatedAttention(
                dim=child.qkv.in_features, num_heads=child.num_heads,
                qkv_bias=child.qkv.bias is not None, qk_scale=child.scale,
                attn_drop=child.attn_drop.p, proj_drop=child.proj_drop.p,
            )
            gated.qkv.load_state_dict(child.qkv.state_dict())
            gated.proj.load_state_dict(child.proj.state_dict())
            # Sync requires_grad
            for p_src, p_dst in [(child.qkv, gated.qkv), (child.proj, gated.proj)]:
                p_dst.weight.requires_grad = p_src.weight.requires_grad
                if p_src.bias is not None:
                    p_dst.bias.requires_grad = p_src.bias.requires_grad
            setattr(model, name, gated)

        elif isinstance(child, CrossAttention) and not isinstance(child, BaseGatedModule):
            gated = GatedCrossAttention(
                dim=child.wq.in_features, num_heads=child.num_heads,
                qkv_bias=child.wq.bias is not None, qk_scale=child.scale,
                attn_drop=child.attn_drop.p, proj_drop=child.proj_drop.p,
            )
            for p in ['wq', 'wk', 'wv', 'proj']:
                getattr(gated, p).load_state_dict(getattr(child, p).state_dict())
                getattr(gated, p).weight.requires_grad = getattr(child, p).weight.requires_grad
                if getattr(child, p).bias is not None:
                    getattr(gated, p).bias.requires_grad = getattr(child, p).bias.requires_grad
            setattr(model, name, gated)

        elif len(list(child.children())):
            add_gates_to_attention(child)


def load_gate_freeze_config(model, json_path):
    """Load freeze masks from JSON. mask=1 → trainable, 0 → frozen.
    JSON format: {"gate_names": ["layer.mu[5]", ...]}"""
    with open(json_path) as f:
        data = json.load(f)

    selected = {}
    for name in data['gate_names']:
        m = re.match(r'(.+)\[(\d+)\]$', name)
        if m:
            selected.setdefault(m.group(1), set()).add(int(m.group(2)))

    for mod_name, mod in model.named_modules():
        if isinstance(mod, GatedConv):
            key = f'{mod_name}.mu'
            channels = selected.get(key, set())
            mask = torch.zeros(mod.gate.shape[0])
            for c in channels:
                mask[c] = 1.0
            mod.gate_mask = mask

        elif isinstance(mod, (GatedAttention, GatedCrossAttention)):
            for gate_type in ['q', 'k', 'v']:
                key = f'{mod_name}.{gate_type}_gate_logit'
                heads = selected.get(key, set())
                mask = torch.zeros(mod.num_heads, 1, 1)
                for h in heads:
                    mask[h] = 1.0
                setattr(mod, f'{gate_type}_gate_mask', mask)


def delta_L(grad, fisher, mode='original'):
    """Compute per-gate-channel Delta L coefficient.

    Contract:
        Type: pure function
        Input: grad [N], fisher [N], mode in {original, exp, inv}
        Output: [N] tensor — per-channel coefficient
        Side effects: none

    mode='original':  -g² / F   (original delta loss formula, negative)
    mode='exp':        exp(-g² / F)  (bounded score in (0, 1])
    mode='inv':        F / g²   (inverted / negative-inverse formula)
    """
    g = grad.reshape(-1)
    f = fisher.reshape(-1)
    if mode == 'exp':
        return torch.exp(-(g ** 2) / (f + 1e-12))
    if mode == 'inv':
        return f / (g ** 2 + 1e-12)
    return -(g ** 2) / (f + 1e-12)


def compute_density_loss(pred, target, den_factor):
    """MSE loss between predicted and target density maps, both scaled.

    Contract:
        Type: pure function
        Input: pred [B,1,H,W] — normalized density map (÷den_factor)
               target [B,1,H,W] — GT Gaussian density (raw, unscaled)
               den_factor — scaling factor (usually 200)
        Output: scalar tensor, MSE(pred * den, target * den)
        Side effects: none
    """
    return torch.nn.functional.mse_loss(
        pred * den_factor, target * den_factor)
