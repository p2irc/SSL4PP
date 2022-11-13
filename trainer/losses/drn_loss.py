from typing import Any, Dict, List, OrderedDict

import torch
from torch import Tensor
from torch.nn import L1Loss

from trainer.registry import LOSSES

from .weighted_smooth_l1_loss import WeightedSmoothL1Loss


def _input_validator(preds: List, targets: List):
    # print(preds)
    # print(targets)
    assert "count" in targets.keys()
    for i in range(1, 6):
        assert i in targets.keys()
    for k in preds.keys():
        assert preds[k].shape == targets[k].shape


@LOSSES.register_class
class DRNLoss:
    def __init__(self, weight: float) -> None:
        self.det_loss = WeightedSmoothL1Loss(weight=weight)
        self.reg_loss = L1Loss()

    def __call__(
        self, output: OrderedDict[Any, Tensor], targets: OrderedDict[Any, Tensor]
    ) -> Dict[str, Tensor]:

        _input_validator(output, targets)
        loss_dict = {}

        for k in output.keys():
            if k == "count":
                loss_dict[k] = self.reg_loss(output[k], targets[k])
            else:
                loss_dict[k] = self.det_loss(output[k], targets[k])
        return loss_dict
