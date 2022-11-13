import torch
import torch.nn as nn
from omegaconf.listconfig import ListConfig

from models.registry import HEADS


@HEADS.register_class
class ConvProjector(nn.Module):
    """Dense projection head using 1x1 convolutions.

    Args:
        layers (ListConfig): A ListConfig object containing information about
            how to build the layers.
    """

    def __init__(self, layers: ListConfig) -> None:
        super().__init__()
        layer_list = []
        for layer in layers:
            layer_list.append(
                nn.Conv2d(
                    layer["in_dim"], layer["out_dim"], 1, bias=layer.get("bias", True)
                )
            )
            if layer.get("batch_norm"):
                layer_list.append(nn.BatchNorm2d(layer["out_dim"]))
            if layer.get("relu"):
                layer_list.append(nn.ReLU(inplace=True))
        self.proj = nn.Sequential(*layer_list)

        # initialize weights -> same as PyTorch implementation of ResNet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch: torch.Tensor):
        """
        from: https://github.com/facebookresearch/vissl/blob/main/vissl/models/heads/mlp.py
        Args:
            batch (torch.Tensor): 4D tensor of shape `N x C x S1 x S2`
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "ConvProjector input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        assert (
            batch.ndim == 4
        ), f"ConvProjector expected 4D tensor of shape NxCxS1xS2. got: {batch.shape}"
        out = self.proj(batch)
        return out
