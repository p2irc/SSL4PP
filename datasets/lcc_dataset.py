from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from datasets.gws_usask import GWSUsask
from datasets.registry import DATASETS

from .file_src_dataset import FileSrcDataset


@DATASETS.register_class
class LCC2017Dataset(GWSUsask):
    """The 2017 Leaf Counting Challenge Dataset.

    Args:
        root: str
            The root directory of the dataset.
        folder: Optional[str]
            The folder of the dataset to use. One of "all", "A1", "A2", "A3" or "A4".
        split: Optional[str]
            The split of the dataset to use. One of "train", or "test".
        subset: Optional[int]
            The subset of the dataset to use. One of 0, 1, 2, 3, 4.
        pyramid_level: Optional[int]
            The pyramid level of the density map.
        transform: Optional[Callable]
            The transformation to apply to the data.
        seed: Optional[int]
            The seed to use for random number generators.

    """

    def __init__(
        self,
        root: str,
        folder: Optional[str] = "all",
        split: Optional[str] = "train",
        subset: Optional[int] = 0,
        pyramid_level: Optional[int] = 3,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:

        root = Path(root).resolve()
        assert root.is_dir(), "Path to dataset root folder is invalid"
        self.root = root
        self.seed = seed
        self.transform = transform

        data_src = self.get_data_src(self.root, split, subset, folder=folder)
        self._samples = self.make_dataset(self.root, data_src)
        self.pyramid_level = pyramid_level

        self.classes = ["leaf"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v, in self.class_to_idx.items()}

    @staticmethod
    def get_data_src(root, split, subset, **kwargs):
        folder = kwargs["folder"]
        assert folder in ["all", "A1", "A2", "A3", "A4"]
        if folder == "all":
            label_src = root.joinpath(f"{split}_{subset}.csv")
        else:
            label_src = root.joinpath(folder, f"{split}_{subset}.csv")
        data_src = pd.read_csv(str(label_src), header=0)
        return data_src

    @staticmethod
    def make_dataset(root: Path, data_src: List):
        file_paths = (
            data_src["rgb_filepath"].apply(lambda x: str(root.joinpath(x))).values
        )
        keypoints = data_src["centers_filepath"].apply(
            lambda x: get_keypoints_from_center_image(str(root.joinpath(x)))
        )
        path_target_pair = list(zip(file_paths, keypoints))
        return path_target_pair


@DATASETS.register_class
class LCC2020Dataset(FileSrcDataset):
    """The 2020 Leaf Counting Challenge Dataset.

    Args:
        root: str
            The root directory of the dataset.
        split: Optional[str]
            The split of the dataset to use. One of "train", "valid", or "val".
        pyramid_level: Optional[int]
            The pyramid level of the density map.
        transform: Optional[Callable]
            The transformation to apply to the data.
        seed: Optional[int]
            The seed to use for random number generators.

    """

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        pyramid_level: Optional[int] = 3,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform, seed)

        assert split in ["train", "valid", "val"]
        if split == "val":
            split = "valid"

        data_src = self.load_data_from_src(self.root, split)
        self._samples = self.make_dataset(self.root, data_src, split)

        self.pyramid_level = pyramid_level

        self.classes = ["leaf"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v, in self.class_to_idx.items()}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def samples(self) -> Tuple[List[str], Any]:
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def make_dataset(
        root: Path, adf: pd.DataFrame, split: str
    ) -> List[Tuple[Any, Any]]:
        """Make the dataset from the data source.

        Args:
            root: Path
                The root directory of the dataset.
            adf: pd.DataFrame
                The data source.
            split: str
                The split of the dataset to use. One of "train", "valid", or "val".

        Returns:
            List[Tuple[Any, Any]]
                The list of samples.

        """
        file_paths = (
            adf["img_id"]
            .apply(lambda x: str(root.joinpath(f"{split}_images", x)))
            .values
        )
        if split == "test":
            keypoints = [[]]
        else:
            keypoints = adf["img_id"].apply(
                lambda x: get_keypoints_from_center_image(
                    str(root.joinpath(f"{split}_images", x.replace("rgb", "centers")))
                )
            )
        path_target_pair = list(zip(file_paths, keypoints))

        return path_target_pair

    @staticmethod
    def load_data_from_src(root: Path, split: str) -> Any:
        """Load the data source.

        Args:
            root: Path
                The root directory of the dataset.
            split: str
                The split of the dataset to use. One of "train", "valid", or "val".

        Returns:
            Any
                The data source.

        """
        src_path = root.joinpath(f"{split}.csv")
        df = pd.read_csv(str(src_path), header=None)
        df.columns = ["img_id", "count"]

        return df

    def __getitem__(self, index: int) -> Any:
        file_path, keypoints = self.samples[index]

        # load image
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(
                image=image, keypoints=keypoints, class_labels=np.zeros(len(keypoints))
            )
            image = transformed["image"]
            keypoints = transformed["keypoints"]

        if len(keypoints) == 0:
            return image

        # generate density map of keypoints
        img_h, img_w = image.shape[:2]

        if self.transform and image.shape[0] <= 3:
            img_h, img_w = image.shape[1:]  # after augmentation, image shape is CxHxW

        target = GWSUsask.get_target(img_h, img_w, keypoints, self.pyramid_level)

        return image, target


def get_keypoints_from_center_image(path_to_img: str) -> np.ndarray:
    """Get the keypoints from the center image."""

    img = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
    keypoints = np.argwhere(img != 0)

    return keypoints
