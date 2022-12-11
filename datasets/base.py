"""The base class for all datasets in the framework."""
import abc
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """The base class for all datasets in the framework.

    Args:
        root: str
            The root directory of the dataset.
        transform: Optional[Callable]:
            The transformation to apply to the data.
        seed: Optional[int] = None
            The seed to use for the random number generator.

    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the dataset."""
        root = Path(root).resolve()
        assert root.is_dir(), "Path to dataset root folder is invalid"
        self.root = root
        self.seed = seed
        self.transform = transform

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """Number of classes in the dataset."""

    @property
    @abc.abstractmethod
    def samples(self) -> List[Tuple[List[str], Any]]:
        """List of samples in the dataset."""

    @property
    def targets(self) -> Any:
        """List of targets in the dataset."""
        return list(zip(*self.samples))[1]

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        """Get sample at given index."""
        raise NotImplementedError
