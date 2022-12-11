"""The TerraByte Field Crop Dataset."""
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import pandas as pd
import PIL
from numpy import ndarray

from datasets.registry import DATASETS

from .file_src_dataset import FileSrcDataset
from .utils import get_random_sample


@DATASETS.register_class
class TFC(FileSrcDataset):
    """The TerraByte Field Crop Dataset.

    Args:
        root: str
            The root directory of the dataset.
        subset: int
            The subset of the dataset to use. One of 1, 2, 3, 4, 5.
        split: Optional[str]
            The split of the dataset to use. One of "train", "val", or "test".
        sample_size: Optional[Union[float, int]]
            The number of samples to use. If float, the number of samples is
            proportional to the size of the dataset. If int, the number of
            samples is exactly the size of the dataset. If None, use all
            samples.
        sampling_method: Optional[str]
            The method to use for sampling. One of "stratified" or "uniform".
        transform: Optional[Callable]
            The transformation to apply to the data.
        seed: Optional[int]
            The seed to use for the random number generator.

    """

    def __init__(
        self,
        root: str,
        subset: int,
        split: Optional[str] = "train",
        sample_size: Optional[Union[float, int]] = None,
        sampling_method: Optional[str] = "stratified",
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the dataset."""
        super().__init__(root, transform, seed)

        self._validate_input(split, subset)

        if split == "train":
            src_path = self.root.joinpath(f"tfc{subset}_{split}.csv")
        else:
            src_path = self.root.joinpath((f"tfc_{split}.csv"))
        data_src = self.load_from_csv(src_path)

        self.classes = data_src["label"].astype("category").cat.categories.values
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v, in self.class_to_idx.items()}

        path_target_pairs = self.make_dataset(self.root, data_src, split)
        self._samples = get_random_sample(
            path_target_pairs, sample_size, sampling_method, self.seed
        )

    @property
    def num_classes(self):
        """Number of classes in the dataset."""
        return len(self.classes)

    @property
    def samples(self) -> Tuple[List[str], Any]:
        """The samples in the dataset."""
        return self._samples

    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.samples)

    @staticmethod
    def _validate_input(split: str, subset: int):
        """Validate the input."""
        assert split in [
            "train",
            "val",
            "test",
        ], "Only train and val splits are supported"
        assert subset in [1, 2, 3, 4, 5]

    @staticmethod
    def make_dataset(
        root: Path, adf: pd.DataFrame, split: str
    ) -> List[Tuple[Any, Any]]:
        """Make the dataset."""
        adf["image_id"] = [
            str(root.joinpath(split, row[1], row[0]))
            for row in adf.itertuples(index=False)
        ]
        adf["label"] = adf["label"].astype("category").cat.codes
        path_target_pairs = list(adf.to_records(index=False))
        return path_target_pairs

    def __getitem__(self, index: int) -> Tuple[ndarray, int]:
        """Get an item from the dataset."""
        file_path, target = self.samples[index]

        # pylint: disable=no-member
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            try:
                transformed = self.transform(image=image)
                # if SSLTransform is used, a list of two images will be returned
                if len(transformed) > 1:
                    image = [
                        transformed_image["image"] for transformed_image in transformed
                    ]
                else:
                    image = transformed["image"]
            except:
                image = self.transform(PIL.Image.fromarray(image))

        return image, target
