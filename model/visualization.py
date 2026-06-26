import os

import torch
import numpy as np


def visualize_density_debug(
    images, pred_dens_dict=None, gt_dens_dict=None,
    cfg_data=None, save_path='debug_vis.png', idx=0, info='',
):
    """
    Overlay density maps on original images for debug.
    Uses FigureCanvasAgg to bypass pyplot (compatible headless / debugger).
    """
    from os.path import dirname, abspath
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Take single image
    img = images[idx] if images.dim() == 4 else images
    assert img.dim() == 3, f'Need (C,H,W), got {img.shape}'

    # Denormalize → [0,1] numpy (H,W,3)
    img_vis = img.cpu().detach().float()
    if cfg_data is not None:
        mean = torch.tensor(cfg_data.MEAN_STD[0], device=img_vis.device).view(3, 1, 1)
        std = torch.tensor(cfg_data.MEAN_STD[1], device=img_vis.device).view(3, 1, 1)
        img_vis = img_vis * std + mean
    img_vis = img_vis.clamp(0, 1).permute(1, 2, 0).numpy()

    def _get_dens(d, key):
        v = d[key]
        v = v[idx] if v.dim() == 4 else v
        return v.cpu().detach().squeeze().numpy()

    src_dict = pred_dens_dict or gt_dens_dict
    keys = list(src_dict.keys())
    n_cols = len(keys) + 1
    n_rows = 1 if gt_dens_dict is None else 2

    fig = Figure(figsize=(5 * n_cols, 5 * n_rows))
    FigureCanvasAgg(fig)
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    for r in range(n_rows):
        axes[r, 0].imshow(img_vis)
        axes[r, 0].set_title('Original' if r > 0 else f'Original\n{info}', fontsize=10)
        axes[r, 0].axis('off')

    row_gt, row_pred = 0, (1 if n_rows == 2 else 0)

    if gt_dens_dict is not None:
        for j, key in enumerate(keys):
            ax = axes[row_gt, j + 1]
            dens = _get_dens(gt_dens_dict, key)
            ax.imshow(img_vis)
            vmax = float(dens.max()) if dens.max() > 0 else 1.0
            ax.imshow(dens, cmap='jet', alpha=0.5, vmin=0, vmax=vmax)
            ax.set_title(f'GT {key}', fontsize=10)
            ax.axis('off')

    if pred_dens_dict is not None:
        for j, key in enumerate(keys):
            ax = axes[row_pred, j + 1]
            dens = _get_dens(pred_dens_dict, key)
            ax.imshow(img_vis)
            vmax = float(dens.max()) if dens.max() > 0 else 1.0
            ax.imshow(dens, cmap='jet', alpha=0.5, vmin=0, vmax=vmax)
            ax.set_title(f'Pred {key}', fontsize=10)
            ax.axis('off')

    fig.tight_layout()
    os.makedirs(dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'[Debug] Visualization saved → {abspath(save_path)}')
