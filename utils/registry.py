"""Registry for building objects from config files."""
import inspect
from functools import partial
from typing import Callable, Dict, Optional, Union

from omegaconf.dictconfig import DictConfig


class Registry:
    """Factory for creating registries.

    This class is used to create registries, which are used to register
    classes. The registries are used to build objects from a config file.

    Args:
        name (str):
            Name of the registry.

    Example:
        >>> BACKBONES = Registry("backbone")
        >>> @BACKBONES.register_class
        >>> class ResNet:
        >>>     pass
        >>> backbone = BACKBONES.get("ResNet")
        >>> assert backbone == ResNet

    """

    def __init__(self, name: str) -> None:
        """Initialize the registry."""
        self._name = name
        self._class_dict = dict()

    @property
    def name(self) -> None:
        """Return the name of the registry."""
        return self._name

    @property
    def class_dict(self) -> None:
        """Return the class dictionary."""
        return self._class_dict

    def get(self, class_name: str) -> Callable:
        """Get the class from the registry.

        Args:
            class_name (str):
                Name of the class to get.

        Returns:
            Callable:
                The class.

        """
        if class_name not in self._class_dict:
            raise KeyError(f"{class_name} is not in the {self._name} registry")
        return self._class_dict[class_name]

    def _register(self, cls: object, force: Optional[bool] = False) -> None:
        """Registers a class.

        Args:
            cls: object
                Class to be registered.
            force: bool, default=False
                If True, overwrites the class if it already exists.

        """
        # check if cls is a class
        if not inspect.isclass(cls):
            raise TypeError("module must be a class, but got {}".format(type(cls)))

        # add class to class_dict, if it doesn't exist already
        class_name = cls.__name__
        if not force and class_name in self._class_dict:
            raise KeyError(
                "{} is already registered in {}".format(class_name, self.name)
            )
        self._class_dict[class_name] = cls

    def register_class(
        self, cls: Optional[Callable] = None, force: Optional[bool] = False
    ) -> Callable:
        """Registers a class.

        Args:
            cls (Callable, optional):
                Class to be registered.
            force (bool, optional):
                If True, overwrites the class if it already exists. Defaults to
                False.

        Returns:
            Callable:
                The registered class.

        """
        if cls is None:
            return partial(self.register_class, force=force)
        self._register(cls, force=force)
        return cls


def build_from_cfg(
    cfg: Union[Dict, DictConfig], registry: object, default_args: Optional[Dict] = None
):
    """A factory method to build a module from config dict.

    Args:
        cfg: dict or DictConfig
            Config dict. It should at least contain the key "type".
        registry: :obj:`Registry`
            The registry to search the type from.
        default_args: dict, optional
            Default initialization arguments.

    Returns:
        obj: The constructed object.

    """
    assert isinstance(cfg, (dict, DictConfig)) and "type" in cfg
    assert isinstance(default_args, dict) or default_args is None

    args = dict(cfg.copy())
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                "{} is not in the {} registry".format(obj_type, registry.name)
            )
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)
