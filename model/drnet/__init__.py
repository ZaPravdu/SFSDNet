"""DRNet model package — Video Individual Counting (CVPR 2022).

Components that reuse SFSDNet's existing implementations:
  * ``model.VGG.conv`` — ResBlock, BasicConv, BasicDeconv
  * ``model.necks.fpn`` — FPN
  * ``misc.layer`` — Gaussianlayer
  * ``model.points_from_den`` — local_maximum_points
  * ``model.PreciseRoIPooling`` — prroi_pool2d
  * ``model.MatchTool`` — associate_pred2gt_point, hungarian
  * ``misc.KPI_pool`` — Task_KPI_Pool
"""

from model.drnet.vic import Video_Individual_Counter
from model.drnet.drnet_model import DRNetModel

__all__ = ['Video_Individual_Counter', 'DRNetModel']
