from pathlib import Path
from typing import Any, Callable, List, Optional, OrderedDict, Tuple

import cv2
import numpy as np

from datasets.registry import DATASETS

from .file_src_dataset import FileSrcDataset
from .utils import generate_density_map


@DATASETS.register_class
class GWSUsask(FileSrcDataset):
    """The Global Wheat Spikelet Usask dataset.

    Args:
        root: str
            The root directory of the dataset.
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
        split: Optional[str] = "train",
        subset: Optional[int] = 0,
        pyramid_level: Optional[int] = 3,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform, seed)

        assert split in ["train", "test"]
        assert subset in [0, 1, 2, 3, 4]
        src_path = self.root.joinpath(f"{split}_{subset}.json")
        data_src = self.load_from_json(src_path)

        self._samples = self.make_dataset(self.root, data_src)

        self.pyramid_level = pyramid_level

        self.classes = ["wheat_spikelet"]
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
    def make_dataset(root: Path, data_src: List) -> List[Tuple[Any, Any]]:
        path_target_pair = []
        for data in data_src:
            filename = root.joinpath(data["image"])
            keypoints = data["annotations"][0]["annotations"]
            path_target_pair.append((str(filename), keypoints))
        return path_target_pair

    @staticmethod
    def get_target(
        img_h: int, img_w: int, keypoints: np.ndarray, pyramid_level: int
    ) -> OrderedDict[Any, Any]:
        """Generate density map of keypoints."""
        map_shape = (np.array([img_h, img_w]) + 2**pyramid_level - 1) // (
            2**pyramid_level
        )

        # rescale keypoints
        ratio = np.array([map_shape[0] / img_h, map_shape[1] / img_w])
        scaled_keypoints = keypoints * ratio

        target = OrderedDict(
            [
                (1, generate_density_map(scaled_keypoints, map_shape, radius=7)),
                (2, generate_density_map(scaled_keypoints, map_shape, radius=7)),
                (3, generate_density_map(scaled_keypoints, map_shape, radius=5)),
                (4, generate_density_map(scaled_keypoints, map_shape, radius=5)),
                (5, generate_density_map(scaled_keypoints, map_shape, radius=3)),
                ("count", len(scaled_keypoints)),
            ]
        )

        return target

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

        assert len(keypoints) > 0

        # generate density map of keypoints
        img_h, img_w = image.shape[:2]

        if self.transform and image.shape[0] <= 3:
            img_h, img_w = image.shape[1:]  # after augmentation, image shape is CxHxW

        target = self.get_target(img_h, img_w, keypoints, self.pyramid_level)

        return image, target
