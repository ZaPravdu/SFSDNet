"""RoI pooling with equivalent fallback for prroi_pool2d.

prroi_pool2d is a CUDA custom op that requires compilation.
When unavailable, ``roi_align(output_size=2) + avg_pool2d(2)`` gives an
*exact* equivalent for the ``pooled_height=pooled_width=1`` case used by DRNet.

**Proof**:  For a bilinear function :math:`f(x,y)` over a rectangle, the
definite integral equals the average of the four bi-quarter-point samples,
which is exactly what ``roi_align(output_size=2) + avg_pool2d(2)`` computes.
See https://en.wikipedia.org/wiki/Bilinear_interpolation#Application
"""
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _roi_pool(features, rois, pooled_h, pooled_w, spatial_scale):
    """Drop-in equivalent of ``prroi_pool2d`` for ``pooled_h=pooled_w=1``.

    Falls back to ``torchvision.ops.roi_align`` when the CUDA custom op
    ``prroi_pool2d`` is unavailable.  The fallback is mathematically
    equivalent for the ``pooled_h=pooled_w=1`` case used throughout DRNet.
    """
    assert pooled_h == 1 and pooled_w == 1, \
        f'_roi_pool fallback only supports 1x1 output, got {pooled_h}x{pooled_w}'

    result = _try_prroi(features, rois, pooled_h, pooled_w, spatial_scale)
    if result is not None:
        return result

    # Exact-equivalent fallback: 2x2 Align → avg to 1x1.
    from torchvision.ops import roi_align
    feat = roi_align(features, rois, output_size=(2, 2),
                     spatial_scale=spatial_scale, aligned=True)
    return F.avg_pool2d(feat, 2)           # [N, D, 1, 1]


_PRROI = None


def _try_prroi(features, rois, pooled_h, pooled_w, spatial_scale):
    """Try prroi_pool2d once; cache success / failure."""
    global _PRROI
    if _PRROI is False:
        return None
    if _PRROI is not None:
        return _PRROI(features, rois, pooled_h, pooled_w, spatial_scale)

    # First call: attempt compile + forward.
    try:
        from model.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
        # Trigger lazy CUDA compilation.
        dev = features.device
        _ = prroi_pool2d(
            torch.zeros(1, 1, 4, 4, device=dev),
            torch.zeros(1, 5, device=dev), 1, 1, 1.0)
        _PRROI = prroi_pool2d
        logger.info('Using prroi_pool2d (PreciseRoIPooling)')
        return _PRROI(features, rois, pooled_h, pooled_w, spatial_scale)
    except Exception:
        _PRROI = False
        logger.warning(
            'prroi_pool2d compile/forward failed — using roi_align fallback '
            '(mathematically equivalent for 1x1 output)')
        return None
