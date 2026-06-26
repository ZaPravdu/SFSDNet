import math
import torch
import torch.nn as nn

from model.gates.base import BaseGatedModule


class BaseGatedAttention(BaseGatedModule):
    """Shared base for GatedAttention and GatedCrossAttention.
    Holds common gate parameters, regularisation, and freeze logic.
    Subclasses implement _project_qkv() and forward()."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Per-head gate logits (H, 1, 1), init 0 → 2*sigmoid(0) = 1.0 (identity)
        self.q_gate_logit = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.k_gate_logit = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.v_gate_logit = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.q_gate_mask = self.k_gate_mask = self.v_gate_mask = None
        self.q_reg_coeff = [1.0] * num_heads
        self.k_reg_coeff = [1.0] * num_heads
        self.v_reg_coeff = [1.0] * num_heads

    def _apply_gates(self, q, k, v):
        return (q * (2 * torch.sigmoid(self.q_gate_logit)),
                k * (2 * torch.sigmoid(self.k_gate_logit)),
                v * (2 * torch.sigmoid(self.v_gate_logit)))

    def _attend(self, q, k, v, B, N):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return (attn @ v).transpose(1, 2).reshape(B, N, -1)

    # ---- Regularisation ----
    def _reg_total(self, fn):
        total = torch.tensor(0., device=self.q_gate_logit.device)
        for name, coeff in [('q', self.q_reg_coeff), ('k', self.k_reg_coeff), ('v', self.v_reg_coeff)]:
            gate = 2 * torch.sigmoid(getattr(self, f'{name}_gate_logit'))
            for h in range(self.num_heads):
                total = total + fn(gate[h] - 1) * coeff[h]
        return total.squeeze()

    def l2_regularization(self):
        return self._reg_total(lambda d: d.pow(2))

    def l1_regularization(self):
        return self._reg_total(lambda d: d.abs())

    def set_reg_coeff(self, q_coeff, k_coeff, v_coeff):
        """Set per-head regularisation coefficients (3 separate lists for Q/K/V)."""
        for name, coeff in [('q', q_coeff), ('k', k_coeff), ('v', v_coeff)]:
            assert len(coeff) == self.num_heads, f"expected {self.num_heads} {name}_coeffs, got {len(coeff)}"
            for c in coeff:
                assert not math.isnan(c)
            setattr(self, f'{name}_reg_coeff', coeff)

    # ---- Freeze mask ----
    def apply_freeze_mask(self):
        for name in ['q', 'k', 'v']:
            mask = getattr(self, f'{name}_gate_mask', None)
            logit = getattr(self, f'{name}_gate_logit', None)
            if mask is not None and logit is not None and logit.grad is not None:
                logit.grad *= mask.to(logit.device)


class GatedAttention(BaseGatedAttention):
    """Self-attention with per-head gates. Fused QKV projection."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = self._apply_gates(q, k, v)
        x = self._attend(q, k, v, B, N)
        x = self.proj(x)
        return self.proj_drop(x)


class GatedCrossAttention(BaseGatedAttention):
    """Cross-attention with per-head gates. Separate Q/K/V projections."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, y):
        B, Nx, C = x.shape
        Ny = y.shape[1]
        q = self.wq(x).reshape(B, Nx, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(y).reshape(B, Ny, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(y).reshape(B, Ny, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k, v = self._apply_gates(q, k, v)
        x = self._attend(q, k, v, B, Nx)
        x = self.proj(x)
        return self.proj_drop(x)


# ── pytest-style tests ──────────────────────────────────────────────────────

def _attn_factory(num_heads=4):
    return GatedAttention(dim=64, num_heads=num_heads)


class TestGatedAttention:
    def test_identity_init(self):
        m = _attn_factory(4)
        x = torch.randn(1, 8, 64)
        out = m(x)
        assert out.shape == (1, 8, 64)

    def test_set_reg_coeff(self):
        m = _attn_factory(4)
        m.set_reg_coeff([.5, 1., 2., 3.], [1.] * 4, [2.] * 4)
        assert m.q_reg_coeff == [.5, 1., 2., 3.]
        assert m.k_reg_coeff == [1., 1., 1., 1.]
        assert m.v_reg_coeff == [2., 2., 2., 2.]

    def test_set_reg_coeff_wrong_length_raises(self):
        m = _attn_factory(4)
        try:
            m.set_reg_coeff([1., 2.], [1., 2., 3., 4.], [1., 2., 3., 4.])
            assert False
        except AssertionError:
            pass

    def test_set_reg_coeff_nan_raises(self):
        m = _attn_factory(4)
        try:
            m.set_reg_coeff([1., 1., float('nan'), 1.], [1.] * 4, [1.] * 4)
            assert False
        except AssertionError:
            pass

    def test_gradient_flows_to_logits(self):
        m = _attn_factory(4)
        x = torch.randn(1, 8, 64)
        y = m(x)
        loss = y.sum()
        loss.backward()
        assert m.q_gate_logit.grad is not None
        assert m.k_gate_logit.grad is not None
        assert m.v_gate_logit.grad is not None

    def test_delta_L_methods_removed(self):
        m = _attn_factory(4)
        assert not hasattr(m, '_init_delta_L_acc')
        assert not hasattr(m, '_accumulate_delta_L')
        assert not hasattr(m, '_finalize_delta_L')
        assert not hasattr(m, '_dl_acc')

    def test_regularisation_scales_with_coeff(self):
        m = _attn_factory(4)
        base = m.l2_regularization().item()
        m.set_reg_coeff([10.] * 4, [10.] * 4, [10.] * 4)
        assert abs(m.l2_regularization().item() - 10.0 * base) < 1e-6


class TestGatedCrossAttention:
    def test_forward_shape(self):
        m = GatedCrossAttention(dim=64, num_heads=4)
        x = torch.randn(1, 8, 64)
        y = torch.randn(1, 12, 64)
        out = m(x, y)
        assert out.shape == (1, 8, 64)

    def test_set_reg_coeff(self):
        m = GatedCrossAttention(dim=64, num_heads=4)
        m.set_reg_coeff([0.1]*4, [0.2]*4, [0.3]*4)
        assert m.q_reg_coeff == [0.1]*4
        assert m.k_reg_coeff == [0.2]*4
