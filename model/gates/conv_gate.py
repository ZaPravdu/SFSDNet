import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gates.base import BaseGatedModule


class GatedConv(BaseGatedModule):
    """Per-channel gated Conv2d: output = conv(x) * 2*sigmoid(gate).

    Two gate modes:
      - ``'independent'`` (default): ``gate`` is a learned per-channel ``nn.Parameter`` (C_out,).
      - ``'input_dependent'``: ``gate`` is an ``nn.Conv2d(C_in, C_out, 3)`` that predicts
        per-channel logits from pooled input features.

    ``self.gate`` is the unified gate attribute in both modes — external code can search
    for ``.gate`` in parameter names without knowing the mode.

    contract:
        Type: nn.Module
        Input: x [B, C_in, H, W]
        Output: [B, C_out, H_out, W_out]
        Side effects: caches gate value in ``self._gate_buffer`` (kept on graph for backward)
    """
    def __init__(self, conv, prior_mean=1.0, gate_mode='independent', **kwargs):
        super().__init__()
        self.conv = conv
        self.gate_mode = gate_mode
        for p in self.conv.parameters():
            p.requires_grad = False

        C_out = conv.out_channels
        self.reg_coeff = [1.0] * C_out
        self._gate_buffer = None

        if gate_mode == 'independent':
            self.gate = nn.Parameter(torch.full((C_out,), float(0.)))
        elif gate_mode == 'input_dependent':
            self.gate = nn.Conv2d(conv.in_channels, C_out, 3)
        else:
            raise ValueError(
                f"Unknown gate_mode='{gate_mode}'. "
                f"Use 'independent' (default) or 'input_dependent'."
            )

        if isinstance(prior_mean, (int, float)):
            prior_mean = torch.full((C_out,), float(prior_mean))
        self.register_buffer('prior_mean', prior_mean)

    def get_gate_logits(self, x=None):
        """Return raw gate logits, shape (B, C_out, 1, 1).

        contract:
            Input: x [B, C_in, H, W] — only read in ``input_dependent`` mode
            Output: gate logits, still on computation graph
        """
        if self.gate_mode == 'independent':
            return self.gate.view(1, -1, 1, 1)
        # input_dependent: pool to 3×3 → conv → (B, C_out, 1, 1)
        return self.gate(F.adaptive_avg_pool2d(x, 3))

    def forward(self, x):
        y = self.conv(x)
        gate_logits = self.get_gate_logits(x)
        gate = 2 * torch.sigmoid(gate_logits)
        self._gate_buffer = gate  # keep graph attached for L1/L2 backward
        return y * gate

    def _reg_total(self, fn):
        if self.gate_mode == 'independent':
            gate = 2 * torch.sigmoid(self.gate)
            coeff = torch.tensor(self.reg_coeff, device=gate.device)
            return (fn(gate - self.prior_mean) * coeff).sum()
        # input_dependent: gate (B, C, 1, 1) from last forward, still on graph
        gate = self._gate_buffer
        coeff = torch.tensor(self.reg_coeff, device=gate.device)
        diff = gate - self.prior_mean.view(1, -1, 1, 1)
        return (fn(diff) * coeff.view(1, -1, 1, 1)).sum()

    def l2_regularization(self):
        return self._reg_total(lambda d: d.pow(2))

    def l1_regularization(self):
        return self._reg_total(lambda d: d.abs())

    def set_reg_coeff(self, coeff_list):
        """Set per-channel regularisation coefficients from external Fisher/Delta-L."""
        n = self.conv.out_channels
        assert len(coeff_list) == n, f"expected {n} coeffs, got {len(coeff_list)}"
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


def _gated_conv_input_dependent_factory(C=4):
    """Build a tiny GatedConv with ``gate_mode='input_dependent'`` for testing."""
    conv = nn.Conv2d(2, C, kernel_size=1)
    return GatedConv(conv, gate_mode='input_dependent')


class TestInputDependentGatedConv:
    """Tests for ``gate_mode='input_dependent'``."""

    def test_identity_init(self):
        """Zero-fill gate.weight/gate.bias → 2*sigmoid(0) = 1.0 → output matches conv(x)."""
        g = _gated_conv_input_dependent_factory(4)
        with torch.no_grad():
            g.gate.weight.zero_()
            if g.gate.bias is not None:
                g.gate.bias.zero_()
        x = torch.randn(1, 2, 8, 8)
        y_plain = g.conv(x)
        y_gated = g(x)
        assert torch.allclose(y_plain, y_gated, atol=1e-6), \
            "zero-filled gate_conv should give identity"

    def test_output_shape(self):
        g = _gated_conv_input_dependent_factory(4)
        x = torch.randn(2, 2, 8, 8)
        y = g(x)
        assert y.shape == (2, 4, 8, 8)

    def test_gradient_flows_to_gate(self):
        """Backward should produce .grad on gate.weight and gate.bias."""
        g = _gated_conv_input_dependent_factory(4)
        x = torch.randn(1, 2, 4, 4)
        y = g(x)
        loss = y.sum()
        loss.backward()
        assert g.gate.weight.grad is not None, "gate.weight should get gradient"
        assert g.gate.weight.grad.abs().sum().item() > 0
        if g.gate.bias is not None:
            assert g.gate.bias.grad is not None
            assert g.gate.bias.grad.abs().sum().item() > 0

    def test_gate_changes_with_input(self):
        """Different inputs → different gate values."""
        g = _gated_conv_input_dependent_factory(4)
        with torch.no_grad():
            g.gate.weight.fill_(0.1)
            g.gate.bias.fill_(0.05)
        x1 = torch.randn(1, 2, 8, 8)
        x2 = 1e6 * torch.randn(1, 2, 8, 8)
        g(x1)
        buf1 = g._gate_buffer.clone()
        g(x2)
        buf2 = g._gate_buffer.clone()
        assert not torch.allclose(buf1, buf2, atol=1e-4), \
            "input-dependent gate should differ for different inputs"

    def test_l2_regularization_scales_with_coeff(self):
        """L2 penalty scales with reg_coeff."""
        g = _gated_conv_input_dependent_factory(4)
        x = torch.randn(2, 2, 8, 8)
        g(x)  # populate _gate_buffer
        base = g.l2_regularization().item()
        g.set_reg_coeff([10.0, 10.0, 10.0, 10.0])
        scaled = g.l2_regularization().item()
        assert abs(scaled - 10.0 * base) < 1e-6, f"expected {10*base}, got {scaled}"

    def test_l1_regularization_returns_scalar(self):
        g = _gated_conv_input_dependent_factory(4)
        x = torch.randn(2, 2, 8, 8)
        g(x)
        reg = g.l1_regularization()
        assert reg.ndim == 0
        assert reg.item() >= 0

    def test_regularization_gradient_flows_to_gate_conv(self):
        """L2 regularization backward should produce grad on gate parameters."""
        g = _gated_conv_input_dependent_factory(4)
        x = torch.randn(2, 2, 8, 8)
        y = g(x)
        reg = g.l2_regularization()
        loss = y.sum() + reg
        loss.backward()
        assert g.gate.weight.grad is not None, \
            "L2 reg should produce gradient on gate.weight"

    def test_set_reg_coeff_wrong_length_raises(self):
        g = _gated_conv_input_dependent_factory(4)
        try:
            g.set_reg_coeff([1.0, 2.0])  # 2 vs 4
            assert False, "should have raised"
        except AssertionError:
            pass

    def test_set_reg_coeff_nan_raises(self):
        g = _gated_conv_input_dependent_factory(4)
        try:
            g.set_reg_coeff([1.0, float('nan'), 1.0, 1.0])
            assert False, "should have raised"
        except AssertionError:
            pass

    def test_gate_buffer_shape(self):
        """After forward, _gate_buffer should be (B, C, 1, 1)."""
        g = _gated_conv_input_dependent_factory(4)
        x = torch.randn(3, 2, 8, 8)
        g(x)
        assert g._gate_buffer.shape == (3, 4, 1, 1)

    def test_gate_buffer_attached(self):
        """_gate_buffer should keep the computation graph (detach not called)."""
        g = _gated_conv_input_dependent_factory(4)
        x = torch.randn(1, 2, 8, 8)
        g(x)
        assert g._gate_buffer.grad_fn is not None, \
            "_gate_buffer should retain grad_fn for the conv graph"

    def test_param_names_contain_gate(self):
        """gate parameters should contain 'gate' for external name-based search."""
        g = _gated_conv_input_dependent_factory(4)
        names = [n for n, _ in g.named_parameters()]
        # Bare names: 'gate.weight' → 'gate' in name works (used by _get_gate_params)
        assert any('gate' in n for n in names), \
            f"no param with 'gate' in name: {names}"
        # Nested names (real model): 'conv1.gate.weight' → '.gate' in name works
        nested = [f'conv1.{n}' for n in names]
        assert any('.gate' in n for n in nested), \
            f"no '.gate' in nested names: {nested}"
