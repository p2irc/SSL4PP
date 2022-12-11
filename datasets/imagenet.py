"""ImageNet dataset."""
from pathlib import Path
from typing import Callable, Optional, Union

from datasets.registry import DATASETS

from .image_dataset_folder import ImageDatasetFolder


@DATASETS.register_class
class ImageNet(ImageDatasetFolder):
    """The ImageNet dataset.

    Args:
        root: str
            The root directory of the dataset.
        split: Optional[str]
            The split of the dataset to use. One of "train", "val".
        sample_size: Optional[Union[int, float]]
            The number of samples to use. If an integer, the exact number of
            samples to use. If a float, the percentage of samples to use. If
            None, use all samples.
        sampling_method: Optional[str]
            The method to use for sampling the dataset. One of "uniform",
            "stratified".
        transform: Optional[Callable]
            The transformation to apply to the data.
        seed: Optional[int]
            The seed to use for the random number generator.

    """

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        sample_size: Optional[Union[float, int]] = None,
        sampling_method: Optional[str] = "stratified",
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the dataset."""
        assert split in ["train", "val"], "Only train and val splits are supported"
        root = str(Path(root).resolve().joinpath(split))
        super().__init__(root, sample_size, sampling_method, transform, seed)
