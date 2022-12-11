"""Projection head for the Dense Contrastive Learning task."""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig

from models.builder import build_head
from models.registry import HEADS


@HEADS.register_class
class ProjectionHead(nn.Module):
    """Combine MLP and Convolutional projection heads in parallell.

    Used for the Dense Contrastive Learning task.

    Args:
        mlp_head (Union[Dict, DictConfig]):
            A Dict or DictConfig object containing information about how to build
            the MLP head.
        dense_head (Optional[Union[Dict, DictConfig]]):
            A Dict or DictConfig object containing information about how to build
            the Dense head.
        grid_size (Optional[int]):
            The size of the grid to use for the Dense head.

    """

    def __init__(
        self,
        mlp_head: Union[Dict, DictConfig],
        dense_head: Optional[Union[Dict, DictConfig]] = None,
        grid_size: Optional[int] = None,
    ) -> None:
        """Init method."""
        super().__init__()

        self.mlp_head = build_head(mlp_head)
        self.dense_head = None

        if dense_head is not None:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.with_pool = grid_size != None
            if self.with_pool:
                self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
            self.dense_head = build_head(dense_head)
            self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor):
        """The forward pass implementation."""
        if self.dense_head is not None:
            if isinstance(x, list):
                assert (
                    len(x) == 1
                ), "DenseProjector input should be either a tensor (3D, 4D) or list containing 1 tensor."
                x = x[0]
            assert (
                x.ndim == 4
            ), f"DenseProjector expected 4D tensor of shape NxCxS1xS2. got: {x.shape}"

            avg_pooled_x = self.avg_pool(x)
            avg_pooled_x = self.mlp_head(avg_pooled_x.view(avg_pooled_x.size(0), -1))

            if self.with_pool:
                x = self.pool(x)  # sxs

            x = self.dense_head(x)  # sxs: nxcxsxs
            avg_pooled_x2 = self.avg_pool2(x)  # 1x1: nxcx1x1
            x = x.view(x.size(0), x.size(1), -1)  # nxcxs^2
            avg_pooled_x2 = avg_pooled_x2.view(avg_pooled_x2.size(0), -1)  # nxc
            return [avg_pooled_x, x, avg_pooled_x2]
        else:
            return self.mlp_head(x)
