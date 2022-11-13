import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
import torch

from datasets.base import Dataset


class FileSrcDataset(Dataset):
    """Object for a dataset that is stored in a file. Supports JSON and CSV
    files.

    Args:
        root: str
            The root directory of the dataset.
        transform: Optional[Callable]
            The transformation to apply to the data.
        seed: Optional[int]
            The seed to use for the random number generator.

    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform, seed)

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes in the dataset."""

    @property
    @abstractmethod
    def samples(self) -> Tuple[List[str], Any]:
        """List of samples in the dataset."""

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        raise NotImplementedError

    @staticmethod
    def load_data_from_src(*args, **kwargs) -> Any:
        """Loads the dataset from the given source."""

    @staticmethod
    def load_from_json(src_path: Path) -> List[dict]:
        """Loads the dataset from a JSON file.

        Args:
            src_path: Path
                The path to the JSON file.

        Returns:
            The dataset as a list of dictionaries.

        """
        assert src_path.exists()
        with open(src_path, "r") as f:
            data_src = json.load(f)

        return data_src

    @staticmethod
    def load_from_csv(src_path) -> pd.DataFrame:
        """Loads the dataset from a CSV file.

        Args:
            src_path: Path
                The path to the CSV file.

        Returns:
            The dataset as a pandas DataFrame.

        """
        assert src_path.exists()

        data_src = pd.read_csv(str(src_path), header=0)
        return data_src

    @staticmethod
    @abstractmethod
    def make_dataset(*args, **kwargs) -> List[Tuple[Any, Any]]:
        """Creates a list of path/target pairs."""

    def __getitem__(self, index: int) -> Any:
        """Get sample at given index."""
        raise NotImplementedError
