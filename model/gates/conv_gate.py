import math
import torch
import torch.nn as nn

from model.gates.base import BaseGatedModule


class GatedConv(BaseGatedModule):
    """Per-channel gated Conv2d: output = conv(x) * 2*sigmoid(gate).
    Accepts **kwargs for backward compat (e.g. mode='independent' from test_gate_core.py)."""
    def __init__(self, conv, prior_mean=1.0, **kwargs):
        super().__init__()
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False

        C = conv.out_channels
        self.reg_coeff = [1.0] * C
        self.gate = nn.Parameter(torch.full((C,), float(0.)))

        if isinstance(prior_mean, (int, float)):
            prior_mean = torch.full((C,), float(prior_mean))
        self.register_buffer('prior_mean', prior_mean)

    def forward(self, x):
        y = self.conv(x)
        g = 2 * torch.sigmoid(self.gate.view(1, -1, 1, 1))
        return y * g

    def _reg_total(self, fn):
        gate = 2 * torch.sigmoid(self.gate)
        total = torch.tensor(0., device=gate.device)
        for i in range(len(gate)):
            total = total + fn(gate[i] - self.prior_mean[i]) * self.reg_coeff[i]
        return total.squeeze()

    def l2_regularization(self):
        return self._reg_total(lambda d: d.pow(2))

    def l1_regularization(self):
        return self._reg_total(lambda d: d.abs())

    def set_reg_coeff(self, coeff_list):
        """Set per-channel regularisation coefficients from external Fisher computation."""
        assert len(coeff_list) == len(self.gate), f"expected {len(self.gate)} coeffs, got {len(coeff_list)}"
        for c in coeff_list:
            assert not math.isnan(c)
        self.reg_coeff = coeff_list


# ── pytest-style tests ──────────────────────────────────────────────────────

def _gated_conv_factory(C=4):
    """Build a tiny GatedConv for testing."""
    conv = nn.Conv2d(2, C, kernel_size=1)
    return GatedConv(conv)


class TestGatedConv:
    def test_identity_init(self):
        """Gate starts at identity: 2*sigmoid(0) = 1.0"""
        g = _gated_conv_factory(4)
        x = torch.randn(1, 2, 4, 4)
        y_plain = g.conv(x)
        y_gated = g(x)
        assert torch.allclose(y_plain, y_gated, atol=1e-6), "gate init should be ≈ identity"

    def test_set_reg_coeff_shape(self):
        g = _gated_conv_factory(4)
        g.set_reg_coeff([0.5, 1.0, 2.0, 3.0])
        assert g.reg_coeff == [0.5, 1.0, 2.0, 3.0]

    def test_set_reg_coeff_wrong_length_raises(self):
        g = _gated_conv_factory(4)
        try:
            g.set_reg_coeff([1.0, 2.0])  # 2 vs 4
            assert False, "should have raised"
        except AssertionError:
            pass

    def test_set_reg_coeff_nan_raises(self):
        g = _gated_conv_factory(4)
        try:
            g.set_reg_coeff([1.0, float('nan'), 1.0, 1.0])
            assert False, "should have raised"
        except AssertionError:
            pass

    def test_regularisation_scales_with_coeff(self):
        """L2 loss should scale with reg_coeff."""
        g = _gated_conv_factory(4)
        base = g.l2_regularization().item()
        g.set_reg_coeff([10.0, 10.0, 10.0, 10.0])
        scaled = g.l2_regularization().item()
        assert abs(scaled - 10.0 * base) < 1e-6, f"expected {10*base}, got {scaled}"

    def test_gradient_flows_to_gate(self):
        """Backward through forward() should produce gate.grad."""
        g = _gated_conv_factory(4)
        x = torch.randn(1, 2, 4, 4)
        y = g(x)
        loss = y.sum()
        loss.backward()
        assert g.gate.grad is not None
        assert g.gate.grad.shape == (4,)
        assert g.gate.grad.abs().sum().item() > 0

    def test_freeze_mask_zeroes_grad(self):
        g = _gated_conv_factory(4)
        x = torch.randn(1, 2, 4, 4)
        y = g(x)
        loss = y.sum()
        loss.backward()
        g.apply_freeze_mask()
        assert g.gate.grad[0].item() == 0.0
        assert g.gate.grad[3].item() == 0.0
        assert g.gate.grad[1].item() != 0.0

    def test_delta_L_methods_removed(self):
        """These methods should NOT exist in the refactored module."""
        g = _gated_conv_factory(4)
        assert not hasattr(g, '_init_delta_L_acc')
        assert not hasattr(g, '_accumulate_delta_L')
        assert not hasattr(g, '_finalize_delta_L')
        assert not hasattr(g, '_dl_acc')
