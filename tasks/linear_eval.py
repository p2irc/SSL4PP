from typing import Any, Optional

import torch
from omegaconf import DictConfig
from wandb.sdk.wandb_run import Run

from tasks.base import Task
from tasks.registry import TASKS
from tasks.classification import ImageClassification
import datasets.builder as dataset_builder


@TASKS.register_class
class LinearEvaluation(ImageClassification):
    """Task for linear evaluation.

    Args:
        cfg: Hydra config object.
        **kwargs: Additional arguments.
    """

    def __init__(self, cfg: DictConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self.train_loader, self.test_loader = dataset_builder.load_datasets(
            self.datasets,
            self.batch_size,
            self.cfg.task.dataloader.num_workers,
            self.cfg.task.dataloader.get("drop_last", True),
            self.cfg.get("seed"),
            test_batch_size=self.batch_size
            // 2,  # NOTE: this is a hack for UWFC dataset; val/test sets too large to fit memory
            collate_fn=kwargs.get("collate_fn"),
        )

    @staticmethod
    def load_pretrained_checkpoint(
        model: torch.nn.Module, url: str, replacement_key: str
    ) -> torch.nn.Module:
        model = Task.load_pretrained_checkpoint(model, url, replacement_key)

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False

        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

        return model

    def train_one_epoch(self, epoch: int, wandb_logger: Optional[Run] = None):
        super().train_one_epoch(epoch, wandb_logger, mode="eval")
