from collections import OrderedDict
from typing import Dict, Tuple, Union

from torch import Tensor
from omegaconf import DictConfig

from tasks.registry import TASKS
from trainer.metrics import CountingMetrics
from tasks.object_detection import ObjectDetection


@TASKS.register_class
class DensityCounting(ObjectDetection):
    """Task for counting via density estimation.

    Args:
        cfg (DictConfig): Hydra config object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        if (
            not cfg.task.model.get("is_pretrained")
            or cfg.checkpoint.get("pretrained") is not None
        ):
            cfg.task.model["is_pretrained"] = (
                cfg.checkpoint.get("pretrained") is not None
            )

        super().__init__(
            cfg, collate_fn=None, state_dict_replacement_key="backbone.body."
        )
        self.test_metrics = CountingMetrics()

    def prepare_input(self, **kwargs) -> Tuple[Tensor]:
        images = self._prepare_images(kwargs["images"])
        return (images,)

    def _prepare_images(self, images):
        return images.cuda(self.device_id, non_blocking=True)

    def _prepare_targets(self, targets):
        return OrderedDict(
            (k, v.unsqueeze(1).cuda(self.device_id)) for k, v in targets.items()
        )

    def get_loss(self, **kwargs) -> Tensor:
        outputs = kwargs["outputs"]
        targets = self._prepare_targets(kwargs["targets"])
        loss_dict = self.criterion(outputs, targets)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    @staticmethod
    def process_results(
        results: Dict[str, Union[Tensor, float]]
    ) -> Dict[str, Union[Tensor, float]]:
        return results
