"""DRNet matching utilities — ROI generation, GT matching, point extraction.

Reuses ``local_maximum_points`` from SFSDNet's
:py:mod:`model.points_from_den` (identical implementation).
"""
import torch

from model.points_from_den import local_maximum_points  # noqa: F401 — re-exported for convenience


class get_ROI_and_MatchInfo:
    """Generate RoI proposals and matching ground-truth from frame pairs.

    Parameters
    ----------
    train_size : (int, int)
        (height, width) of training images.
    radius : int
        Half-side of the square RoI window around each point (pixels).
    feature_scale : float
        Ratio between feature-map spatial size and input spatial size
        (e.g. 1/4 for DRNet's 256-dim descriptors).
    """
    def __init__(self, train_size: tuple[int, int],
                 radius: int = 8, feature_scale: float = 0.25):
        assert len(train_size) == 2
        assert radius > 0
        self.h, self.w = train_size
        self.radius = radius
        self.feature_scale = feature_scale

    def __call__(self, target_a: dict, target_b: dict,
                 noise: str | None = None,
                 shape: tuple[int, int] | None = None):
        """Build ROIs and match GT for a frame pair.

        Parameters
        ----------
        target_a, target_b : dict
            Each must have ``points`` (Nx2) and ``person_id`` (N).
        noise : str | None
            ``'ab'`` — add Gaussian noise (sigma=2) to both frames.
            ``'a'`` — noise only frame A (sigma=1).
            ``'b'`` — noise only frame B (sigma=1).
            ``None`` — no noise (validation / test).
        shape : (int, int) | None
            Override image spatial size (h, w).  Used when the effective
            shape differs from ``train_size`` (e.g. variable-size inference).

        Returns
        -------
        match_gt : dict
            ``a2b`` — matched pair indices  (M, 2).
            ``un_a`` — unmatched indices in frame A.
            ``un_b`` — unmatched indices in frame B.
        pois : Tensor  [N0+N1, 5]
            Concatenated ROIs ``[batch_id, x1, y1, x2, y2]``.
        """
        assert 'points' in target_a and 'person_id' in target_a
        assert 'points' in target_b and 'person_id' in target_b

        gt_a = target_a['points']
        gt_b = target_b['points']

        h, w = (shape[0], shape[1]) if shape is not None else (self.h, self.w)

        # Optional noise injection (training augmentation).
        if noise == 'ab':
            gt_a = gt_a + torch.randn_like(gt_a) * 2
            gt_b = gt_b + torch.randn_like(gt_b) * 2
        elif noise == 'a':
            gt_a = gt_a + torch.randn_like(gt_a)
        elif noise == 'b':
            gt_b = gt_b + torch.randn_like(gt_b)
        elif noise is not None:
            raise ValueError(f'Unknown noise mode: {noise}')

        def _roi(points, batch_id, h, w):
            r = self.radius
            roi = points.new_zeros(len(points), 5)
            roi[:, 0] = batch_id
            roi[:, 1] = (points[:, 0] - r).clamp(min=0)
            roi[:, 2] = (points[:, 1] - r).clamp(min=0)
            roi[:, 3] = (points[:, 0] + r).clamp(max=w)
            roi[:, 4] = (points[:, 1] + r).clamp(max=h)
            return roi

        pois = torch.cat([
            _roi(gt_a, 0, h, w),
            _roi(gt_b, 1, h, w),
        ], dim=0)

        # Build match GT from person_id cross-reference.
        a_ids = target_a['person_id']
        b_ids = target_b['person_id']
        dist = (a_ids.unsqueeze(1) - b_ids.unsqueeze(0)).abs()
        matched_a, matched_b = torch.where(dist == 0)

        match_gt = {
            'a2b': torch.stack([matched_a, matched_b], dim=1),
            'un_a': torch.where(dist.min(1)[0] > 0)[0],
            'un_b': torch.where(dist.min(0)[0] > 0)[0],
        }

        return match_gt, pois
