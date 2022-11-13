import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import PIL
import numpy as np

from datasets.base import Dataset
from .utils import get_random_sample


class ImageDatasetFolder(Dataset):
    """A generic dataset that can load images with a folder structure like:
    root/class_name/img.ext

        Args:
            root: str
                The root directory of the dataset
            sample_size: Optional[Union[float, int]] = None
                The number of samples to be used in the dataset. If float, it is
                the percentage of samples to use. If int, it is the exact number
                of samples to use.
            sampling_method: Optional[str] = "stratified"
                The sampling method to use. One of "stratified", "uniform".
            transform: Optional[Callable] = None
                The transformation to apply to the data.
            seed: Optional[int] = None
                The seed to use for the random number generator.

        Attributes:
            classes: List[str]
                The list of classes in the dataset.
            class_to_idx: Dict[str, int]
                The mapping from class name to class index.
            idx_to_class: Dict[int, str]
                The mapping from class index to class name.
            samples: List[Tuple[List[str], Any]]
                The list of samples in the dataset. Each sample is a tuple of
                (file_path, target).
    """

    IMG_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".JPEG",
        ".png",
        ".PNG",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

    def __init__(
        self,
        root: str,
        sample_size: Optional[Union[float, int]] = None,
        sampling_method: Optional[str] = "stratified",
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform, seed)

        self._validate_input(sampling_method, sample_size)

        self.classes, self.class_to_idx = self.find_classes(self.root)
        self.idx_to_class = {v: k for k, v, in self.class_to_idx.items()}
        path_target_pair = self.make_dataset(self.root, self.class_to_idx)
        self._samples = get_random_sample(
            path_target_pair, sample_size, sampling_method, self.seed
        )

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def samples(self) -> Tuple[List[str], Any]:
        return self._samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
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

    @staticmethod
    def _validate_input(
        sampling_method: str, sample_size: Union[float, int], *args, **kwargs
    ) -> None:
        """Check if the input is valid."""
        assert sampling_method in ["stratified", "uniform"], (
            "Only stratified "
            "random sampling and uniform random sampling are supported."
        )
        assert (
            sample_size is None
            or isinstance(sample_size, (float, int))
            and sample_size > 0
        )

    @staticmethod
    def find_classes(root: Path) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        return classes, class_to_idx

    @staticmethod
    def make_dataset(root: Path, class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
        path_target_pair = []
        for ext in ImageDatasetFolder.IMG_EXTENSIONS:
            filepaths = list(root.glob(f"*{os.sep}*{ext}"))

            if not filepaths:
                continue

            # extract the class name from the file path
            target_class = list(map(lambda x: x.parent.name, filepaths))
            target_idx = [class_to_idx[item] for item in target_class]

            filepath_str = list(map(lambda x: str(x), filepaths))
            path_target_pair.extend(zip(filepath_str, target_idx))

        if len(path_target_pair) == 0:
            warnings.warn(
                f"No files with extension in {ImageDatasetFolder.IMG_EXTENSIONS} was found."
            )

        return path_target_pair
