import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import pandas as pd
from omegaconf import DictConfig
from wandb.sdk.wandb_run import Run
from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .base import Task
from tasks.registry import TASKS
from utils.logger import MetricLogger
from trainer.metrics import AverageDomainAccuracy


@TASKS.register_class
class ObjectDetection(Task):
    """Task for object detection.

    Args:
        cfg (DictConfig): Hydra config object.

    Attributes:
        best_eval_score (float): Best evaluation score.
        is_best_ckpt (bool): Whether the current checkpoint is the best.
    """

    def __init__(self, cfg: DictConfig, **kwargs: Any) -> None:
        if (
            not isinstance(cfg.task.model.get("is_pretrained"), bool)
            or cfg.checkpoint.get("pretrained") is not None
        ):
            cfg.task.model["is_pretrained"] = (
                cfg.checkpoint.get("pretrained") is not None
            )

        collate_func = kwargs.get("collate_fn", collate_fn)
        super().__init__(
            cfg, collate_fn=collate_func, state_dict_replacement_key="backbone.body."
        )

        metrics = [MeanAveragePrecision(compute_on_cpu=True)]
        if "GWHD" in type(self.datasets["test"]).__name__:
            metrics.append(AverageDomainAccuracy(compute_on_cpu=True))

        # on torchmetrics v0.8.0: compute_on_cpu prevents DDP gather_all operation.
        # move metrics to GPU and do computation on GPU (this is much slower)
        if self.in_dist_mode:
            for metric in metrics:
                metric.cuda(self.device_id)
                metric.compute_on_cpu = False
        self.test_metrics = MetricCollection(metrics)

    def prepare_input(self, **kwargs) -> Tuple:
        images = kwargs["images"]
        targets = kwargs["targets"]
        if torch.cuda.is_available():
            images = self._prepare_images(images)
            targets = self._prepare_targets(targets)
        return (images, targets)

    def _prepare_images(self, images):
        return list(img.cuda(self.device_id, non_blocking=True) for img in images)

    def _prepare_targets(self, targets):
        return [
            {k: v.cuda(self.device_id, non_blocking=True) for k, v in t.items()}
            for t in targets
        ]

    def get_loss(self, **kwargs) -> Union[float, torch.Tensor]:
        loss_dict = kwargs["outputs"]
        loss = sum(loss for loss in loss_dict.values())
        return loss

    def get_train_metrics(self, *args, **kwargs) -> None:
        return None

    @torch.no_grad()
    def evaluate(self, wandb_logger: Optional[Run] = None, **kwargs):
        print("Evaluating model...")
        eval_time = time.time()
        self.model.eval()

        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"

        current_step = (kwargs["epoch"] + 1) * len(self.train_loader)
        for images, targets in metric_logger.log_every(
            self.test_loader, self.cfg.log_freq, header
        ):
            if torch.cuda.is_available():
                images = self._prepare_images(images)
                targets = self._prepare_targets(targets)

            model_time = time.time()
            outputs = self.model(images)

            self.test_metrics.update(outputs, targets)
            metric_logger.update(model_time=time.time() - model_time)

        # compute performance metrics and unpack results
        eval_results = self.process_results(self.test_metrics.compute())

        # print results
        for k, v in eval_results.items():
            print("{:<15} {:<10}".format(k, v))

        if wandb_logger:
            wandb_logger.log(
                data={
                    "epoch": kwargs["epoch"] + 1,
                    **{
                        f"{self.cfg.task.dataset.test_split.split}/{k}": v
                        for k, v in eval_results.items()
                    },
                },
                step=current_step,
            )

        # reset state for the next epoch
        if self.test_metrics is not None:
            self.test_metrics.reset()

        print("Done. Total time for evalutation: ", time.time() - eval_time)

        return eval_results

    @staticmethod
    def process_results(
        results: Dict[str, Union[torch.Tensor, float]]
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Process the results from the metrics.

        Args:
            results (Dict[str, Union[torch.Tensor, float]]): Results from the metrics.

        Returns:
            Dict[str, Union[torch.Tensor, float]]: Processed results.
        """
        eval_results = dict()
        for name, value in results.items():
            if name == "AverageDomainAccuracy":
                eval_results["ADA"] = value
            else:
                value = round(value.item(), 4)
                if value != -1:
                    eval_results[name] = value
        return eval_results


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate
    function (to be passed to the DataLoader).

    Args:
        batch: an iterable of N sets from __getitem__()

    Returns:
        a tensor of images, lists of varying-size tensors of bounding boxes,
        labels, and difficulties
    """

    images = list()
    targets = list()

    for i, t in batch:
        images.append(i)
        targets.append(t)

    return images, targets
