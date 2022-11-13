import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import PIL
from numpy import ndarray

from .file_src_dataset import FileSrcDataset
from .image_dataset_folder import ImageDatasetFolder
from .registry import DATASETS
from .utils import get_random_sample


@DATASETS.register_class
class INat2021(FileSrcDataset):
    """The 2021 iNaturalist dataset.

    Args:
        root: str
            The root directory of the dataset.
        split: Optional[str]
            The split of the dataset to use. One of "train", "train_mini", "val", "test".
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
        split: Optional[str] = "train_mini",
        supercategory: Optional[str] = None,
        sample_size: Optional[Union[float, int]] = None,
        sampling_method: Optional[str] = "stratified",
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform, seed)

        self._validate_input(sampling_method, sample_size, split, supercategory)

        src_path = self.root.joinpath(f"{split}.json")
        metadata = self.load_from_json(src_path)

        self.classes, self.class_to_idx = self.find_classes(metadata, supercategory)
        self.idx_to_class = {v: k for k, v, in self.class_to_idx.items()}
        img_target_pairs = self.make_dataset(
            self.root, metadata, self.class_to_idx, supercategory
        )
        self._samples = get_random_sample(
            img_target_pairs, sample_size, sampling_method, self.seed
        )

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def samples(self) -> Tuple[List[str], Any]:
        return self._samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[ndarray, int]:
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
        sampling_method: str,
        sample_size: Union[int, float],
        split: str,
        supercategory: str,
    ) -> None:
        ImageDatasetFolder._validate_input(sampling_method, sample_size)
        assert split in [
            "train",
            "train_mini",
            "val",
            "public_test",
        ], f"{split} split is not supported"
        assert (
            supercategory
            in [
                "Birds",
                "Amphibians",
                "Mammals",
                "Mollusks",
                "Reptiles",
                "Ray-finned Fishes",
                "Arachnids",
                "Plants",
                "Fungi",
                "Animalia",
                "Insects",
            ]
            or supercategory is None
        ), (
            f"{supercategory} is " "not defined for the iNaturlist 2021 dataset"
        )

    @staticmethod
    def find_classes(
        metadata: Dict, supercategory: str
    ) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class labels in the data.

        Args:
            metadata: Dict
                The metadata of the dataset.
            supercategory: Optional[str]
                The supercategory to use for the dataset. If None, use all
                supercategories.

        Returns:
            classes: List[str]
                The class labels in the data.
            class_to_idx: Dict[str, int]
                A mapping from class label to class index.

        """
        categories = metadata["categories"]

        classes = []
        for i in categories:
            if supercategory is None or i["supercategory"] == supercategory:
                # get the class name
                classes.append(i["name"].replace(" ", "_"))

        classes = sorted(classes)  # just to make sure
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        return classes, class_to_idx

    @staticmethod
    def make_dataset(
        root: Path, metadata: Dict, class_to_idx: Dict[str, int], supercategory: str
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of the form (image path, class index).

        Args:
            root: Path
                The root directory of the dataset.
            metadata: Dict
                The metadata of the dataset.
            class_to_idx: Dict[str, int]
                A mapping from class label to class index.
            supercategory: Optional[str]
                The supercategory to use for the dataset. If None, use all
                supercategories.

        Returns:
            samples: List[Tuple[str, int]]
                A list of samples of the form (image path, class index).

        """
        instances = []
        annotations = metadata.get("annotations")
        if annotations is not None:  # train, val or train_mini
            for idx, item in enumerate(annotations):
                cat = metadata["categories"][item["category_id"]]  # get category data
                if supercategory is None or cat["supercategory"] == supercategory:
                    img_data = metadata["images"][idx]
                    filename = img_data["file_name"]
                    path = root.joinpath(filename)

                    # get class index
                    cls_name = cat["name"].replace(" ", "_")
                    cls_idx = class_to_idx[cls_name]
                    instances.append((str(path), cls_idx))
        else:  # public test dataset
            for img in metadata["images"]:
                # get the full path, which is expected by __get_item()__
                filename = img["file_name"]
                path = root.joinpath(filename)
                instances.append((str(path), 0))
                warnings.warn(
                    "The targets for this dataset have all been set to 0!", UserWarning
                )
        return instances
