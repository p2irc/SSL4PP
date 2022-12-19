"""Weighted Smooth L1 Loss."""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import SmoothL1Loss

from trainer.registry import LOSSES


@LOSSES.register_class
class WeightedSmoothL1Loss(SmoothL1Loss):
    """Weighted Smooth L1 Loss.

    Args:
        reduce: str, optional
            Specifies the reduction to apply to the output: 'none' | 'mean' |
            'sum'.
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements
            in the output,
            'sum': the output will be summed.
        beta: float, default=1
            Specifies the beta parameter for the SmoothL1Loss formulation.
        weight: float, default=0.1
            The weight to use for the positive class.

    """

    def __init__(
        self,
        reduce=None,
        reduction: Optional[str] = "mean",
        beta: Optional[float] = 1,
        weight: Optional[float] = 0.1,
    ) -> None:
        """Init method."""
        super().__init__(reduce, reduction, beta)
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Forward method."""
        l1_loss = F.smooth_l1_loss(pred, target, reduction="none", beta=self.beta)
        weight_complement = torch.tensor(
            1 - self.weight, dtype=pred.dtype, device=pred.device
        )
        weight = torch.ones_like(pred, device=pred.device) * self.weight
        weights = torch.where(pred > 0, weight_complement, weight)
        loss = torch.sum((weights * l1_loss), dim=(2, 3))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
