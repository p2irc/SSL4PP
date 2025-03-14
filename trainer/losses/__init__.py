"""Losses."""
import inspect

from torch import nn

from trainer.registry import LOSSES

from .contrastive_loss import ContrastiveLoss, DenseCLLoss
from .drn_loss import DRNLoss
from .weighted_smooth_l1_loss import WeightedSmoothL1Loss

nn_modules = inspect.getmembers(nn, inspect.isclass)
for cls in nn_modules:
    if "Loss" in cls[0]:
        LOSSES.register_class(cls[1])
LOSSES.register_class(nn.CosineSimilarity)
