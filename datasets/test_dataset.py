from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2

from datasets.base import Dataset
from .utils import get_random_sample


class TestDataset(Dataset):
    """A generic dataset for loading images in a folder.

    Args:
        root: str
            The root directory of the dataset
        sample_size: Optional[Union[float, int]] = None
            The number of samples to be used in the dataset. If float, it is
            the percentage of samples to use. If int, it is the exact number
            of samples to use.
        transform: Optional[Callable] = None
            The transformation to apply to the data.
        seed: Optional[int] = None
            The seed to use for the random number generator.
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
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform, seed)
        paths = [
            (filepath,)
            for ext in TestDataset.IMG_EXTENSIONS
            for filepath in list(self.root.glob(f"*{ext}"))
        ]
        self._samples = get_random_sample(paths, sample_size, "uniform", self.seed)

    @property
    def num_classes(self) -> int:
        return -1

    @property
    def samples(self) -> List[Tuple[List[str]]]:
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        file_path = str(self.samples[index][0])

        # pylint: disable=no-member
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            # if SSLTransform is used, a list of two images will be returned
            if len(transformed) > 1:
                image = [
                    transformed_image["image"] for transformed_image in transformed
                ]
            else:
                image = transformed["image"]

        return (image, index)
