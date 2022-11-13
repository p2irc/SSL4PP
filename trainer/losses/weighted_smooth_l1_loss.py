from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import SmoothL1Loss

from trainer.registry import LOSSES


@LOSSES.register_class
class WeightedSmoothL1Loss(SmoothL1Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: Optional[str] = "mean",
        beta: Optional[float] = 1,
        weight: Optional[float] = 0.1,
    ) -> None:
        super().__init__(size_average, reduce, reduction, beta)
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
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
