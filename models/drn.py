""" Detection and Regression Network (DRN) from
https://www.frontiersin.org/articles/10.3389/fpls.2021.575751/full

Hacked together by Franklin Ogidi
"""
from collections import OrderedDict
from typing import Optional, OrderedDict, Tuple

import torch
from torch import nn
from torch import Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone,
    _validate_trainable_layers,
)

from omegaconf.dictconfig import DictConfig

from models.registry import MODELS


@MODELS.register_class
class DRN(nn.Module):
    """Detection and Regression Network. Adapted from:
    https://www.frontiersin.org/articles/10.3389/fpls.2021.575751/full

    Args:
        backbone (DictConfig): Config for the backbone, has to contain:
            - type (str): Key to registry.
        num_classes (int): Number of classes to predict.
        is_pretrained (bool): Whether to load pretrained weights.
        trainable_backbone_layers (int): Number of trainable layers in the backbone.
        spatial_nms_kernel_size (int): Size of the spatial NMS kernel.
        spatial_nms_stride (int): Stride of the spatial NMS kernel.
        spatial_nms_beta (float): Beta parameter for spatial NMS.
        smooth_step_func_thresh (Tuple[float, float]): Thresholds for the smooth step function.
        smooth_step_func_betas (Tuple[float, float]): Betas for the smooth step function.

    Returns:
        OrderedDict[str, Tensor]: Heatmaps and count.
    """

    def __init__(
        self,
        backbone: DictConfig,
        num_classes: Optional[int] = 1,
        is_pretrained: Optional[bool] = False,
        trainable_backbone_layers: Optional[int] = None,
        spatial_nms_kernel_size: Optional[int] = 3,
        spatial_nms_stride: Optional[int] = 1,
        spatial_nms_beta: Optional[float] = 100,
        smooth_step_func_thresh: Tuple[float, float] = (0.4, 0.8),
        smooth_step_func_betas: Tuple[float, float] = (1, 15),
    ) -> None:
        super().__init__()
        assert backbone["type"] == "ResNet", "Only the ResNet backbone is supported."
        assert backbone["depth"] in [18, 34, 50, 101, 152]
        self.num_classes = num_classes

        trainable_backbone_layers = _validate_trainable_layers(
            is_pretrained, trainable_backbone_layers, 5, 3
        )

        self.backbone = resnet_fpn_backbone(
            backbone_name=f"{backbone['type'].lower()}{backbone['depth']}",
            pretrained=False,  # don't load the default ImageNet weights
            returned_layers=[2, 3, 4],  # don't return P2
            trainable_layers=trainable_backbone_layers,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d
            if is_pretrained
            else nn.BatchNorm2d,
        )
        self.det_head = DetectionHead(self.backbone.out_channels, num_classes)
        self.count_reg = RegressionHead(
            num_classes,
            spatial_nms_kernel_size,
            spatial_nms_stride,
            spatial_nms_beta,
            smooth_step_func_thresh,
            smooth_step_func_betas,
        )

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        features = self.backbone(images)  # returns P3, P4, P5, pool
        feat4, heat_maps = self.det_head(features["0"])
        count = self.count_reg(feat4, heat_maps[-1])
        out = OrderedDict(zip(range(1, 6), heat_maps))
        out["count"] = count
        return out


class DetectionHead(nn.Module):
    """Detection head for DRN."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.mid_out1 = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, 1), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.mid_out2 = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, 1), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.mid_out3 = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, 1), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.mid_out4 = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, 1), nn.ReLU(inplace=True)
        )
        self.final_heat_map = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1), nn.ReLU(inplace=True)
        )

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv2(feat2)
        feat4 = self.conv2(feat3)
        final_heat_map = self.final_heat_map(feat4)

        out1 = self.mid_out1(feat1)
        out2 = self.mid_out2(feat2)
        out3 = self.mid_out3(feat3)
        out4 = self.mid_out4(feat4)

        return feat4, (out1, out2, out3, out4, final_heat_map)


class RegressionHead(nn.Module):
    """Regression head for DRN."""

    def __init__(
        self,
        num_classes: int,
        spatial_nms_kernel_size: int = 3,
        spatial_nms_stride: int = 1,
        spatial_nms_beta: float = 100,
        smooth_step_func_thresh: Tuple[float, float] = (0.4, 0.8),
        smooth_step_func_betas: Tuple[float, float] = (1, 15),
    ) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.smooth_step_func1 = SmoothStepFunction(
            smooth_step_func_thresh[0], smooth_step_func_betas[0]
        )
        self.smooth_step_func2 = SmoothStepFunction(
            smooth_step_func_thresh[1], smooth_step_func_betas[1]
        )
        self.spatial_nms = SpatialNMS(
            spatial_nms_kernel_size, spatial_nms_stride, spatial_nms_beta
        )
        self.gsp = GlobalSumPool2D()
        self.fc = nn.LazyLinear(out_features=num_classes)

    def forward(self, conv4_out: Tensor, final_heat_map: Tensor) -> Tensor:
        gap_feat = self.gap(conv4_out)  # BxCx1x1
        gap_feat = torch.flatten(gap_feat, 1)

        det_count = self.smooth_step_func1(final_heat_map)
        det_count = self.spatial_nms(det_count)
        det_count = self.smooth_step_func2(det_count)
        det_count = self.gsp(det_count)  # count from heat map; shape: Bx1

        fc_in = torch.cat([gap_feat, det_count], dim=-1)  # shapeBxC+1x1
        count = self.fc(fc_in)  # shape: Bx1

        return count


class SpatialNMS(nn.Module):
    """Spatial Non-Maximum Suppression module."""

    def __init__(self, kernel_size: int = 3, stride: int = 1, beta: float = 1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding=1)
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        p = x
        q = self.maxpool(p)
        abs_p_sub_q = torch.abs(torch.subtract(p, q))
        out = p * torch.exp(-abs_p_sub_q * self.beta)
        return out


class SmoothStepFunction(nn.Module):
    """Smooth step function module."""

    def __init__(self, threshold: float = 0.8, beta: float = 15):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid((x - (torch.ones_like(x) * self.threshold)) * self.beta)


class GlobalSumPool2D(nn.Module):
    """Global sum pooling module."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=(2, 3))
