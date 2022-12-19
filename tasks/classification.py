"""Image classification task."""
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torchmetrics import Accuracy, F1Score, MetricCollection
from wandb.sdk.wandb_run import Run

import utils.distributed as dist_utils
from tasks.registry import TASKS
from utils.logger import MetricLogger

from .base import Task


@TASKS.register_class
class ImageClassification(Task):
    """Task for image classification.

    Args:
        cfg (DictConfig):
            Config object.

    Attributes:
        best_eval_score (float):
            Best evaluation score.
        is_best_ckpt (bool):
            Whether the current checkpoint is the best.

    """

    def __init__(self, cfg: DictConfig, **kwargs: Any) -> None:
        """Init method."""
        super().__init__(cfg)
        metric_dict = {
            "top1_acc": Accuracy(),
            "top5_acc": Accuracy(top_k=5),
            "f1_score": F1Score(
                num_classes=self.datasets["train"].num_classes, average="weighted"
            ),
        }
        if self.cfg.task.model.num_classes <= 5:
            metric_dict.pop("top5_acc")
        if torch.cuda.is_available():
            metric_dict = {k: v.cuda(self.device_id) for k, v in metric_dict.items()}
        metrics = MetricCollection(metric_dict)
        self.train_metrics = metrics.clone(prefix="train/")
        self.test_metrics = metrics.clone(
            prefix=self.cfg.task.dataset.test_split.split + "/"
        )
        self.best_eval_score = 0

    def prepare_input(self, **kwargs) -> Tuple:
        """Prepare the input for the model."""
        images = kwargs["images"].cuda(self.device_id, non_blocking=True)
        return (images,)

    def get_loss(self, **kwargs) -> Union[float, torch.Tensor]:
        """Get the loss."""
        outputs = kwargs["outputs"]
        targets = (
            kwargs["targets"]
            .type(torch.LongTensor)
            .cuda(device=self.device_id, non_blocking=True)
        )
        loss = self.criterion(outputs, targets)
        return loss

    def get_train_metrics(self, **kwargs) -> Dict[str, Any]:
        """Get the metrics for training."""
        predicitions = kwargs["preds"]
        targets = (
            kwargs["targets"]
            .type(torch.LongTensor)
            .cuda(device=self.device_id, non_blocking=True)
        )
        metrics = self.train_metrics(
            torch.nn.functional.softmax(predicitions, dim=-1), targets
        )
        return metrics

    def evaluate(self, wandb_logger: Optional[Run] = None, **kwargs):
        """Evaluate the model."""
        # switch to evaluate mode
        self.model.eval()

        self.is_best_ckpt = False

        metric_logger = MetricLogger(delimiter="  ")
        header = "Test: "

        current_step = (kwargs["epoch"] + 1) * len(self.train_loader)
        print_freq = self.cfg.log_freq
        with torch.inference_mode():
            for images, targets in metric_logger.log_every(
                self.test_loader, print_freq, header
            ):
                if torch.cuda.is_available():
                    images = images.cuda(self.device_id, non_blocking=True)
                    targets = targets = targets.type(torch.LongTensor).cuda(
                        device=self.device_id, non_blocking=True
                    )

                # compute output
                output = self.model(images)

                # compute loss
                loss = self.criterion(output, targets)
                metric_logger.update(loss=loss.item())

                # compute performance metrics
                metrics = self.test_metrics(
                    torch.nn.functional.softmax(output, dim=-1), targets
                )
                for key, value in metrics.items():
                    metric_logger.meters[key].update(value.item())

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        eval_name = self.cfg.task.dataset.test_split.split
        top1_score = metric_logger.meters[f"{eval_name}/top1_acc"].global_avg
        if top1_score > self.best_eval_score:
            self.best_eval_score = top1_score
            self.is_best_ckpt = True

        if wandb_logger:
            wandb_logger.log(
                data={
                    f"{eval_name}/loss": metric_logger.loss.global_avg,
                    **{key: metric_logger.meters[key].global_avg for key in metrics},
                },
                step=current_step,
            )

            if self.is_best_ckpt:
                wandb_logger.summary["top1_acc"] = metric_logger.meters[
                    f"{eval_name}/top1_acc"
                ].global_avg

        print({key: metric_logger.meters[key].global_avg for key in metrics})

        # reset state for the next epoch
        self.test_metrics.reset()

    def save_on_master(
        self, epoch: int, keep_latest_only: bool = True, **kwargs: Any
    ) -> None:
        """Save the model on the master process.

        Args:
            epoch (int): Current epoch.
            keep_latest_only (bool): Whether to keep only the latest checkpoint.
            kwargs (Any): Additional arguments.

        """
        super().save_on_master(epoch, keep_latest_only, **kwargs)

        # save the best model
        if dist_utils.is_main_process() and keep_latest_only and self.is_best_ckpt:
            # get latest checkpoint
            checkpoint_dir = Path(
                self.cfg.checkpoint.get("dir", "checkpoints")
            ).resolve()
            latest_ckpt = checkpoint_dir.joinpath("latest.pt").resolve()
            best_ckpt = checkpoint_dir.joinpath("best_ckpt.pt")

            # copy and rename
            shutil.copy(latest_ckpt, best_ckpt)
