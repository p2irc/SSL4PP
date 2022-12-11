"""Metrics for counting tasks."""
from typing import Dict, List, Optional, OrderedDict

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score


def _input_validator(
    preds: OrderedDict[str, Tensor], targets: OrderedDict[str, Tensor]
):
    """Ensure the correct input format of `preds` and `targets`."""
    if not isinstance(preds, OrderedDict):
        raise ValueError("Expected argument `preds` to be of type OrderedDict")
    if not isinstance(targets, OrderedDict):
        raise ValueError("Expected argument `target` to be of type OrderedDict")
    if len(preds) != len(targets):
        raise ValueError(
            "Expected argument `preds` and `target` to have the same length"
        )

    for k in [1, 2, 3, 4, 5, "count"]:
        if k not in preds.keys():
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")
        if k not in targets.keys():
            raise ValueError(f"Expected all dicts in `target` to contain the `{k}` key")

    if any(type(item) is not Tensor for item in preds.values()):
        raise ValueError("Expected all items in `preds` to be of type Tensor")
    if any(type(item) is not Tensor for item in targets.values()):
        raise ValueError("Expected all items in `target` to be of type Tensor")


class CountingMetrics(Metric):
    """Metrics for counting tasks.

    Args:
        compute_on_step: bool
            Whether to compute the metrics on each step or not.

    Attributes:
        groundtruth_counts: List[Tensor]
            The groundtruth counts.
        predicted_counts: List[Tensor]
            The predicted counts.

    """

    groundtruth_counts: List[Tensor]
    predicted_counts: List[Tensor]

    def __init__(self, compute_on_step: Optional[bool] = False, **kwargs) -> None:
        """Init method."""
        super().__init__(compute_on_step=compute_on_step, **kwargs)

        self.add_state("groundtruth_counts", default=[], dist_reduce_fx=None)
        self.add_state("predicted_counts", default=[], dist_reduce_fx=None)

    def update(
        self, preds: OrderedDict[str, Tensor], target: OrderedDict[str, Tensor]
    ) -> None:
        """Update the metric state."""
        _input_validator(preds, target)

        self.predicted_counts.extend(
            preds["count"].round()
        )  # round to the nearest integer
        self.groundtruth_counts.extend(target["count"])

    def compute(self) -> Dict[str, float]:
        """Compute the metrics."""
        self.groundtruth_counts = torch.as_tensor(self.groundtruth_counts)
        self.predicted_counts = torch.as_tensor(self.predicted_counts)

        metrics = {
            "countAgreement": 100
            * torch.eq(self.predicted_counts, self.groundtruth_counts)
            .type(torch.float)
            .mean(),
            "countDiff": torch.sub(
                self.groundtruth_counts, self.predicted_counts
            ).mean(),
            "rmse": mean_squared_error(
                self.predicted_counts, self.groundtruth_counts, False
            ),
            "mse": mean_squared_error(self.predicted_counts, self.groundtruth_counts),
            "mae": mean_absolute_error(self.predicted_counts, self.groundtruth_counts),
            "r2_score": r2_score(self.predicted_counts, self.groundtruth_counts),
        }
        return metrics
