from typing import Any, Optional

import torch.nn as nn
import torchvision.models.detection
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone,
    _validate_trainable_layers,
)
from omegaconf.dictconfig import DictConfig

from models.registry import MODELS


@MODELS.register_class
class FasterRCNN(torchvision.models.detection.FasterRCNN):
    """Faster R-CNN model. Adapted from torchvision.models.detection.FasterRCNN."""

    def __init__(
        self,
        backbone: DictConfig,
        num_classes: int,
        is_pretrained: Optional[bool] = False,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
    ):
        assert backbone.type == "ResNet", "Only the ResNet backbone is supported."
        assert backbone.depth in [18, 34, 50, 101, 152]

        trainable_backbone_layers = _validate_trainable_layers(
            is_pretrained, trainable_backbone_layers, 5, 3
        )

        backbone_with_fpn = resnet_fpn_backbone(
            f"{backbone.type.lower()}{backbone.depth}",
            pretrained=False,  # don't load the default ImageNet weights
            trainable_layers=trainable_backbone_layers,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d
            if is_pretrained
            else nn.BatchNorm2d,
        )

        super().__init__(backbone_with_fpn, num_classes, **kwargs)
