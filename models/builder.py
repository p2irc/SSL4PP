"""Factory for building models and their components."""
import sys

sys.path.append("..")
from typing import Dict, Optional

from omegaconf import DictConfig
from torch import nn

from utils.registry import Registry, build_from_cfg

from .registry import BACKBONES, HEADS, MODELS


def build(cfg: DictConfig, registry: Registry, default_args: Optional[Dict] = None):
    """Build a module.

    Args:
        cfg (DictConfig):
            Config for the module. Must contain key ``type``.
        registry (Registry):
            Registry to search the type from.
        default_args (dict, optional):
            Default arguments to pass to `cfg`.

    Returns:
        nn.Module:
            The constructed module.

    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg: DictConfig):
    """Build backbone.

    Args:
        cfg (DictConfig):
            Config for the backbone. Must contain key ``type``.

    Returns:
        nn.Module:
            The constructed backbone.

    """
    return build(cfg, BACKBONES)


def build_head(cfg: DictConfig):
    """Build head.

    Args:
        cfg (DictConfig):
            Config for the head. Must contain key ``type``.

    Returns:
        nn.Module:
            The constructed head.

    """
    return build(cfg, HEADS)


def build_model(cfg: DictConfig):
    """Build model.

    Args:
        cfg (DictConfig):
            Config for the model. Must contain key ``type``.

    Returns:
        nn.Module:
            The constructed model.

    """
    return build(cfg, MODELS)
