"""
Gate variance estimation for per-scene regularisation coefficients.

Pipeline:
  1. collect_per_scene_gates() — fine-tune gates on each scene independently
  2. compute_reg_coeff_from_variance() — cross-scene variance -> 1/variance
  3. estimate_gate_uncertainty() — convenience wrapper for step 1 + 2

Each scene starts from identity gates (2*sigmoid(0) = 1.0), so the
gate value variance across scenes measures scene sensitivity
(high variance = domain-specific = uncertain = small reg_coeff).
"""

from typing import Dict, List

import torch
import torch.nn as nn

# Used only in tests for building models with real gate modules
from model.gate_utils import add_gates_to_conv  # noqa: F401
from tqdm import tqdm


def collect_per_scene_gates(
    student_model: nn.Module,
    per_scene_loaders: Dict[str, torch.utils.data.DataLoader],
    lr: float,
    den_factor: float = 200.0,
    epochs: int = 1,
    device: str = "cuda",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Fine-tune gates on each scene independently, collect gate values on CPU.

    After each scene, the collected values are detached and moved to CPU,
    then all gate parameters are reset to 0.0 (identity) for the next scene.

    Args:
        student_model: Video_Counter with identity gates injected,
                       non-gate params frozen (requires_grad=False).
        per_scene_loaders: dict[scene_name, DataLoader] from get_per_scene_loaders().
        lr: learning rate for gate-only AdamW optimiser.
        den_factor: density map scaling factor (cfg_data.DEN_FACTOR).
        epochs: number of epochs per scene.
        device: torch device.

    Returns:
        dict[scene_name, dict[param_name, 1-D tensor on CPU]]
        e.g. {'scene_8': {'Extractor.backbone.0.conv1.gate': tensor([...]), ...},
              'scene_9': {...}}
    """
    gate_params = [
        (n, p) for n, p in student_model.named_parameters()
        if p.requires_grad and ('gate' in n or 'gate_logit' in n)
    ]
    if not gate_params:
        raise RuntimeError("No trainable gate parameters found — did you inject gates?")
    assert per_scene_loaders, (
        "collect_per_scene_gates called with empty per_scene_loaders"
    )

    student_model = student_model.to(device).train()
    criterion = nn.MSELoss()

    per_scene_gates: Dict[str, Dict[str, torch.Tensor]] = {}

    for scene_name, loader in per_scene_loaders.items():
        print(f"[gate_variance] Fine-tuning scene: {scene_name}  "
              f"samples: {len(loader.dataset)}")

        optimizer = torch.optim.AdamW(
            [p for _, p in gate_params], lr=lr, weight_decay=0,
        )

        steps = 0
        for _ in range(epochs):
            for batch in tqdm(loader):
                if batch[0] is None or batch[0].numel() == 0:
                    continue

                weak_imgs, _, targets = batch
                weak_imgs = weak_imgs.to(device)

                out = student_model(weak_imgs, targets)
                global_loss = criterion(out[0] * den_factor, out[1].detach() * den_factor)
                share_loss = criterion(out[2] * den_factor, out[3].detach() * den_factor)
                io_loss = criterion(out[4] * den_factor, out[5].detach() * den_factor)
                loss = (global_loss + 10 * share_loss + io_loss) / 3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps += 1

        print(f"  -> {steps} steps, collecting gate values")

        gate_vals: Dict[str, torch.Tensor] = {}
        for name, param in gate_params:
            val = (2 * torch.sigmoid(param)).detach().cpu().reshape(-1)
            gate_vals[name] = val
        per_scene_gates[scene_name] = gate_vals

        # Reset gate parameters to identity for next scene
        for _, param in gate_params:
            param.data.fill_(0.0)

    return per_scene_gates


def compute_reg_coeff_from_variance(
    per_scene_gate_values: Dict[str, Dict[str, torch.Tensor]],
    eps: float = 1e-12,
) -> Dict[str, List[float]]:
    """
    Cross-scene variance -> reg_coeff = 1 / variance  (ridge regression).

    For each gate parameter, stacks values across all scenes,
    computes population variance per channel/head, returns its reciprocal.

    Args:
        per_scene_gate_values: from collect_per_scene_gates().
        eps: small constant to avoid division-by-zero.

    Returns:
        dict[param_name, list[float]] — reg_coeff per channel/head.
        Ready to be passed to _apply_external_reg_coeff().
    """
    scene_names = list(per_scene_gate_values.keys())
    if len(scene_names) < 2:
        raise ValueError(
            f"Need at least 2 scenes for variance, got {len(scene_names)}"
        )

    param_names = list(per_scene_gate_values[scene_names[0]].keys())
    reg_coeff: Dict[str, List[float]] = {}

    for pname in param_names:
        stacked = torch.stack(
            [per_scene_gate_values[s][pname] for s in scene_names], dim=0,
        )
        var = stacked.var(dim=0, unbiased=False)
        coeff = 1.0 / (var + eps)
        reg_coeff[pname] = coeff.tolist()

    return reg_coeff


def estimate_gate_uncertainty(
    student_model: nn.Module,
    per_scene_loaders: Dict[str, torch.utils.data.DataLoader],
    lr: float,
    den_factor: float = 200.0,
    epochs: int = 1,
    eps: float = 1e-12,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """
    Full pipeline: per-scene fine-tuning -> variance -> reg_coeff dict.

    Returns dict[param_name, list[float]] directly usable by
    P2RModel._apply_external_reg_coeff().
    """
    gate_values = collect_per_scene_gates(
        student_model, per_scene_loaders, lr, den_factor, epochs, device,
    )
    return compute_reg_coeff_from_variance(gate_values, eps)


# ── pytest-style tests ──────────────────────────────────────────────────────


class TestComputeRegCoeffFromVariance:
    """Pure-function tests for compute_reg_coeff_from_variance()."""

    def test_basic_variance_to_coeff(self):
        """Known variance → expected 1/var coeff."""
        gate_values = {
            'scene_A': {'conv.gate': torch.tensor([1.0, 2.0, 3.0, 4.0])},
            'scene_B': {'conv.gate': torch.tensor([2.0, 3.0, 4.0, 5.0])},
            'scene_C': {'conv.gate': torch.tensor([3.0, 4.0, 5.0, 6.0])},
        }
        # Per-channel means: [2, 3, 4, 5]
        # Variance (pop): [2/3, 2/3, 2/3, 2/3] = [0.666..., 0.666..., 0.666..., 0.666...]
        # Coeff = 1/(var+eps) ≈ [1.5, 1.5, 1.5, 1.5]
        result = compute_reg_coeff_from_variance(gate_values, eps=0.0)
        key = 'conv.gate'
        assert key in result
        for i, c in enumerate(result[key]):
            assert abs(c - 1.5) < 1e-5, f"channel {i}: expected 1.5, got {c}"

    def test_zero_variance_channels(self):
        """All-gates-identical across scenes → var=0 → coeff=1/eps (large but bounded)."""
        gate_values = {
            's1': {'g0.gate': torch.tensor([0.5, 1.0])},
            's2': {'g0.gate': torch.tensor([0.5, 1.0])},
        }
        result = compute_reg_coeff_from_variance(gate_values, eps=1e-12)
        # Both channels have var=0 → coeff ≈ 1/eps ≈ 1e12
        for c in result['g0.gate']:
            assert abs(c - 1e12) / 1e12 < 1e-3, f"expected ~1e12, got {c}"

    def test_attention_gate_logit(self):
        """Per-head gate_logit variance works correctly."""
        gate_values = {
            's1': {'attn.q_gate_logit': torch.tensor([0.8, 1.2])},
            's2': {'attn.q_gate_logit': torch.tensor([1.2, 0.8])},
        }
        # var per head: [(0.8-1.0)²+(1.2-1.0)²]/2 = 0.04, same for head 1
        # coeff = 1/0.04 = 25
        result = compute_reg_coeff_from_variance(gate_values, eps=0.0)
        for c in result['attn.q_gate_logit']:
            assert abs(c - 25.0) < 1e-5

    def test_only_one_scene_raises(self):
        """Fewer than 2 scenes should raise."""
        gate_values = {'s1': {'g.gate': torch.tensor([1.0])}}
        import pytest
        with pytest.raises(ValueError, match='at least 2 scenes'):
            compute_reg_coeff_from_variance(gate_values)

    def test_conserves_param_order(self):
        """Parameter keys in output match input order."""
        gate_values = {
            's1': {
                'a.gate': torch.tensor([1.0]),
                'b.gate': torch.tensor([2.0]),
                'c.gate': torch.tensor([3.0]),
            },
            's2': {
                'a.gate': torch.tensor([1.5]),
                'b.gate': torch.tensor([2.5]),
                'c.gate': torch.tensor([3.5]),
            },
        }
        result = compute_reg_coeff_from_variance(gate_values, eps=0.0)
        assert list(result.keys()) == ['a.gate', 'b.gate', 'c.gate']

    def test_channel_mismatch_across_scenes_raises(self):
        """Different tensor shapes across scenes → KeyError on stack."""
        gate_values = {
            's1': {'g.gate': torch.tensor([1.0, 2.0])},
            's2': {'g.gate': torch.tensor([1.0])},  # only 1 channel
        }
        import pytest
        with pytest.raises(RuntimeError):
            compute_reg_coeff_from_variance(gate_values)


class _MiniModel(nn.Module):
    """Tiny model matching Video_Counter's forward(imgs, targets) -> 7-tuple."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.conv2 = nn.Conv2d(4, 1, 1)

    def forward(self, imgs, targets=None):
        x = self.conv1(imgs)
        x = self.conv2(x)
        return (x, x, x, x, x, x, {})


class TestCollectPerSceneGatesReset:
    """Behavioural test: verify gate reset and collection correctness."""

    def _make_model_with_gates(self):
        """Build a tiny model with real GatedConv, all non-gate weights frozen."""
        model = _MiniModel()
        for p in model.parameters():
            p.requires_grad = False
        add_gates_to_conv(model)
        return model

    def test_reset_to_identity_between_scenes(self):
        """After scene 1 fine-tuning and reset, gates should be ≈ identity (2*sigmoid(0)=1.0)."""
        model = self._make_model_with_gates()
        device = 'cpu'

        gate_params = [(n, p) for n, p in model.named_parameters()
                       if p.requires_grad and 'gate' in n]
        assert len(gate_params) > 0, "model should have gate params"

        # Record initial gate values
        initial = {n: (2 * torch.sigmoid(p)).clone() for n, p in gate_params}

        # Fake loaders: 2 scenes, 2 batches each
        from collections import namedtuple
        FakeDataset = namedtuple('FakeDataset', ['dataset'])

        fake_loaders = {}
        for scene in ['scene_P', 'scene_Q']:
            batches = []
            for _ in range(2):
                weak = torch.randn(2, 3, 8, 8)
                strong = torch.randn(2, 3, 8, 8)
                targets = [{}, {}]
                batches.append((weak, strong, targets))

            class _IterLoader:
                def __init__(self, bs):
                    self.bs = bs
                    self.dataset = list(range(len(bs)))  # len(loader.dataset)
                def __iter__(self):
                    return iter(self.bs)

            fake_loaders[scene] = _IterLoader(batches)

        result = collect_per_scene_gates(
            model, fake_loaders, lr=0.01, den_factor=200.0, epochs=1, device=device,
        )

        # Verify reset: after all scenes, gates should be back to identity
        for name, param in gate_params:
            current = (2 * torch.sigmoid(param)).detach()
            assert torch.allclose(current, initial[name], atol=1e-6), \
                f"{name} was not reset to identity"

        # Verify collection has all scenes and params
        assert set(result.keys()) == {'scene_P', 'scene_Q'}
        first_scene = list(result.keys())[0]
        for name, _ in gate_params:
            assert name in result[first_scene], \
                f"missing gate param {name} in collected values"

    def test_collected_values_on_cpu(self):
        """All collected gate values should be on CPU."""
        model = self._make_model_with_gates()
        device = 'cpu'

        class _IterLoader:
            def __init__(self):
                self.dataset = [None]  # len > 0
            def __iter__(self):
                weak = torch.randn(2, 3, 8, 8)
                strong = torch.randn(2, 3, 8, 8)
                targets = [{}, {}]
                return iter([(weak, strong, targets)])

        fake_loaders = {'scene_X': _IterLoader(), 'scene_Y': _IterLoader()}

        result = collect_per_scene_gates(
            model, fake_loaders, lr=0.001, den_factor=200.0, epochs=1, device=device,
        )

        for scene in result.values():
            for name, tensor in scene.items():
                assert tensor.device == torch.device('cpu'), \
                    f"{name} is on {tensor.device}, expected cpu"

    def test_no_gate_params_raises(self):
        """Model without gate params should raise."""
        model = nn.Conv2d(3, 4, 1)  # plain conv, no gates
        for p in model.parameters():
            p.requires_grad = False

        import pytest
        with pytest.raises(RuntimeError, match='No trainable gate'):
            collect_per_scene_gates(model, {}, lr=0.001, device='cpu')
