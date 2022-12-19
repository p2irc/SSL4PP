"""Task for counting via density estimation."""
from collections import OrderedDict
from typing import Dict, Tuple, Union

from omegaconf import DictConfig
from torch import Tensor

from tasks.object_detection import ObjectDetection
from tasks.registry import TASKS
from trainer.metrics import CountingMetrics


@TASKS.register_class
class DensityCounting(ObjectDetection):
    """Task for counting via density estimation.

    Args:
        cfg: DictConfi
            A Hydra config object.

    """

    def __init__(self, cfg: DictConfig) -> None:
        """Init method."""
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
        """Prepare the input for the model."""
        images = self._prepare_images(kwargs["images"])
        return (images,)

    def _prepare_images(self, images):
        """Prepare the images for the model."""
        return images.cuda(self.device_id, non_blocking=True)

    def _prepare_targets(self, targets):
        """Prepare the targets for the model."""
        return OrderedDict(
            (k, v.unsqueeze(1).cuda(self.device_id)) for k, v in targets.items()
        )

    def get_loss(self, **kwargs) -> Tensor:
        """Get the loss."""
        outputs = kwargs["outputs"]
        targets = self._prepare_targets(kwargs["targets"])
        loss_dict = self.criterion(outputs, targets)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    @staticmethod
    def process_results(
        results: Dict[str, Union[Tensor, float]]
    ) -> Dict[str, Union[Tensor, float]]:
        """Process the results."""
        return results
