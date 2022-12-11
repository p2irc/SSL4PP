"""Multi-Layer Perceptron projection head."""
import torch
import torch.nn as nn
from omegaconf.listconfig import ListConfig

from models.registry import HEADS


@HEADS.register_class
class MLP(nn.Module):
    """Multi-Layer Perceptron projection head.

    Args:
        layers (ListConfig):
            A ListConfig object containing information about how to build the
            layers.

    """

    def __init__(self, layers: ListConfig) -> None:
        """Init method."""
        super().__init__()
        layer_list = []
        for layer in layers:
            layer_list.append(
                nn.Linear(
                    layer["in_dim"], layer["out_dim"], bias=layer.get("bias", True)
                )
            )
            if layer.get("batch_norm"):
                layer_list.append(nn.BatchNorm1d(layer["out_dim"]))
            if layer.get("relu"):
                layer_list.append(nn.ReLU(inplace=True))
        self.fc = nn.Sequential(*layer_list)

        # initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(
                module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)
            ):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, batch: torch.Tensor):
        """Forward pass implementation.

        from: https://github.com/facebookresearch/vissl/blob/main/vissl/models/heads/mlp.py

        """
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "MLP input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        if batch.ndim > 2:
            assert all(
                d == 1 for d in batch.shape[2:]
            ), f"MLP expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
            batch = batch.reshape((batch.size(0), batch.size(1)))
        out = self.fc(batch)
        return out
