from collections import OrderedDict
from typing import Any

import torch
from torchvision.models import resnet

from models.registry import BACKBONES, MODELS


@MODELS.register_class
@BACKBONES.register_class
class ResNet(resnet.ResNet):
    """ResNet backbone.

    Args:
        depth (int): Depth of the ResNet backbone.
        **kwargs: Other arguments.
    """

    arch_params = {
        18: (resnet.BasicBlock, [2, 2, 2, 2]),
        34: (resnet.BasicBlock, [3, 4, 6, 3]),
        50: (resnet.Bottleneck, [3, 4, 6, 3]),
        101: (resnet.Bottleneck, [3, 4, 23, 3]),
        152: (resnet.Bottleneck, [3, 8, 36, 3]),
    }

    def __init__(self, depth: int, **kwargs: Any) -> None:
        if depth not in self.arch_params:
            raise KeyError("invalid depth {} for resnet".format(depth))
        block = self.arch_params[depth][0]
        layers = self.arch_params[depth][1]
        super().__init__(block, layers, **kwargs)
        self._fhooks = []
        self._activations = OrderedDict()

        for name, module in list(self.named_modules()):
            # get the output of each Block
            if "layer" in name and "relu" in name:
                self._fhooks.append(
                    module.register_forward_hook(self.forward_hook(name))
                )

    @property
    def activations(self):
        """Returns the activations of the last forward pass."""
        return self._activations

    def forward_hook(self, layer_name):
        """Returns a forward hook for the given layer name."""

        def hook(module, input, output):
            self._activations[layer_name] = output.detach()

        return hook

    def _forward_impl(self, x):
        """Forward pass implementation."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if not isinstance(self.avgpool, torch.nn.Identity):
            x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
