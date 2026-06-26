"""
Fisher Information Matrix — diagonal on model output, with mean gradients.

Fisher (of output μ, no targets involved):
    F_i = E_x[ Σ_p (∂μ_p / ∂θ_i)² ]

Mean gradient (of per-pixel squared error L_p = (μ_p - t_p)²):
    mg_i = E_x[ (1/P) · Σ_p ∂L_p / ∂θ_i ]

The output Fisher is estimated via Monte Carlo random projection so that only
O(mc_iters) backward passes are needed instead of O(H × W):

    ε_p ~ N(0,1),  v = Σ_p ε_p · μ_p
    ∂v/∂θ_i = Σ_p ε_p · ∂μ_p/∂θ_i
    E[(∂v/∂θ_i)²] = Σ_p (∂μ_p/∂θ_i)²      ← unbiased estimate
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def compute_fisher(
    forward_fn: Callable,
    loader,
    params: List[Tuple[str, torch.Tensor]],
    device: torch.device,
    mc_iters: int = 10,
    max_samples: Optional[int] = None,
    quiet: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute diagonal Fisher of model output + mean gradient of per-pixel MSE.

    Per sample: forward_fn(batch) → (mu, target) | None
        mu:      model prediction [B, C, H, W] (on device, requires-graph attached)
        target:  GT or pseudo-label, same shape, detached

    Returns
    -------
    fisher:    dict name→tensor — F_i = E[ Σ_p (∂μ_p / ∂θ_i)² ]
    grad_mean: dict name→tensor — mg_i = E[ ∂L_p / ∂θ_i ]
    """
    fisher = {name: torch.zeros_like(p, device="cpu") for name, p in params}
    grad_mean = {name: torch.zeros_like(p, device="cpu") for name, p in params}
    n = 0

    it = list(loader)
    if not quiet:
        from tqdm import tqdm
        it = tqdm(it, desc="Fisher", leave=False)

    for bidx, batch in enumerate(it):
        if max_samples is not None and n >= max_samples:
            break
        result = forward_fn(batch)
        if result is None:
            continue
        mu, target = result

        # 1) Mean gradient of per-pixel squared error L = (μ - t)²  (exact)
        L = (mu - target).pow_(2)
        torch.autograd.backward(L.sum(), retain_graph=(mc_iters > 0))
        for name, p in params:
            if p.grad is not None:
                grad_mean[name] += p.grad.detach().cpu() 
            p.grad = None

        # 2) Fisher of μ via Monte Carlo  (no targets)
        for mc in range(mc_iters):
            eps = torch.randn_like(mu)
            torch.autograd.backward(
                (eps * mu).sum(),
                retain_graph=(mc < mc_iters - 1),
            )
            for name, p in params:
                if p.grad is not None:
                    # E[g²] = Σ_p (∂μ_p/∂θ)²  (sum over all output positions, single sample)
                    fisher[name] += p.grad.detach().cpu().pow_(2) / mc_iters
                p.grad = None

        n += 1

    if n == 0:
        raise RuntimeError("compute_fisher: no samples processed (forward_fn returned None for all)")

    for name in fisher:
        fisher[name] /= n
        grad_mean[name] /= n
    return fisher, grad_mean


# ── pytest-style tests ──────────────────────────────────────────────────────


class TestComputeFisher:
    """Verify compute_fisher with single-sample frames (matching project convention: input[0]=2 frames)."""

    def test_linear_weight_fisher(self):
        """
        Single-sample linear: μ = Wx,  x ∈ ℝ^{d_in},  μ ∈ ℝ^{d_out}
        ∂μ_j / ∂W_ji = x_i   (only the output channel j that W_ji connects to)
        F[W_ji] = E[ Σ_j (∂μ_j/∂W_ji)² ] = E[ x_i² ]
        Per sample: F[W_ji] = x_i² (independent of output channel j).
        """
        torch.manual_seed(42)
        d_in, d_out = 3, 2
        lin = nn.Linear(d_in, d_out, bias=False)
        x = torch.randn(d_in)                      # one sample, vector
        expected = x.pow(2)                         # [d_in]

        params = [(n, p) for n, p in lin.named_parameters()]

        def fn(batch):
            mu = lin(batch[0])                         # [d_out]
            return mu, torch.zeros_like(mu)

        fisher, _ = compute_fisher(fn, [(x,)], params, 'cpu', mc_iters=500, quiet=True)
        W_name = params[0][0]
        for j in range(d_out):
            for i in range(d_in):
                assert abs(fisher[W_name][j, i].item() - expected[i].item()) < expected[i].item() * 0.25, \
                    f"F[W_{j},{i}]: expected {expected[i].item():.4f}, got {fisher[W_name][j,i].item():.4f}"

    def test_bias_fisher_is_one(self):
        """
        μ_j = (Wx)_j + b_j   →   ∂μ_j / ∂b_j = 1
        Σ_j (∂μ_j / ∂b_j)² = 1  per sample.
        """
        torch.manual_seed(42)
        lin = nn.Linear(4, 3, bias=True)
        x = torch.randn(4)

        params = [(n, p) for n, p in lin.named_parameters() if 'bias' in n]

        def fn(batch):
            mu = lin(batch[0])
            return mu, torch.zeros_like(mu)

        fisher, _ = compute_fisher(fn, [(x,)], params, 'cpu', mc_iters=500, quiet=True)
        b_name = params[0][0]
        for j in range(3):
            assert abs(fisher[b_name][j].item() - 1.0) < 0.25, \
                f"F[b_{j}]: expected 1.0, got {fisher[b_name][j].item():.4f}"

    def test_fisher_and_grad_are_different(self):
        """Fisher (of μ) and grad_mean (of MSE loss) are fundamentally different quantities."""
        torch.manual_seed(42)
        lin = nn.Linear(4, 3)
        x = torch.randn(4)
        y = torch.randn(3)

        params = [(n, p) for n, p in lin.named_parameters()]

        def fn(batch):
            mu = lin(batch[0])
            return mu, batch[1]

        fisher, grad_mean = compute_fisher(fn, [(x, y)], params, 'cpu', mc_iters=50, quiet=True)
        for name in fisher:
            f_flat = fisher[name].flatten()
            g_flat = grad_mean[name].flatten()
            assert f_flat.abs().mean().item() > 1e-8, f"Fisher for {name} should be non-zero"
            assert not torch.allclose(f_flat, g_flat.pow(2), atol=1e-4), \
                f"Fisher and grad_mean² should not match for {name}"

    def test_skip_batch_returns_none(self):
        """Batches where forward_fn returns None should be silently skipped."""
        torch.manual_seed(42)
        lin = nn.Linear(4, 2)

        params = [(n, p) for n, p in lin.named_parameters()]

        def fn(batch):
            if batch.get('skip', False):
                return None
            mu = lin(batch['x'])
            return mu, torch.zeros_like(mu)

        loader = [{'x': torch.randn(4), 'skip': True},
                  {'x': torch.randn(4), 'skip': False}]
        fisher, _ = compute_fisher(fn, loader, params, 'cpu', mc_iters=5, quiet=True)
        assert all(v.abs().sum().item() > 0 for v in fisher.values())

    def test_positive_definite(self):
        """All Fisher diagonal entries should be non-negative (squared values)."""
        torch.manual_seed(7)
        lin = nn.Linear(3, 2)
        x = torch.randn(3)

        params = [(n, p) for n, p in lin.named_parameters()]

        def fn(batch):
            mu = lin(batch[0])
            return mu, torch.zeros_like(mu)

        fisher, _ = compute_fisher(fn, [(x,)], params, 'cpu', mc_iters=30, quiet=True)
        for name in fisher:
            assert (fisher[name] >= 0).all(), f"Fisher for {name} has negative entries"
