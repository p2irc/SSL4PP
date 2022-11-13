from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from numpy import ndarray

from .image_dataset_folder import ImageDatasetFolder
from .file_src_dataset import FileSrcDataset
from .utils import get_random_sample
from datasets.registry import DATASETS


@DATASETS.register_class
class GWHD2021(FileSrcDataset):
    """The 2021 Global Wheat Head Detection dataset.

    Args:
        root: str
            The root directory of the dataset.
        split: Optional[str]
            The split of the dataset to use. One of "train", "val", "test".
        sample_size: Optional[Union[int, float]]
            The number of samples to use. If an integer, the exact number of
            samples to use. If a float, the percentage of samples to use.
        transform: Optional[Callable]
            The transformation to apply to the data.
        seed: Optional[int]
            The seed to use for the random number generator.
    """

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        sample_size: Optional[Union[int, float]] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform, seed)

        src_path = self.root.joinpath(f"competition_{split}.csv")
        data_src = self.load_from_csv(src_path)

        domains = data_src["domain"].astype("category").cat.codes.values
        # print(data_src["domain"].astype("category").cat.categories.values) # DEBUG

        path_target_pair = self.make_dataset(self.root, data_src)
        data = list(zip(path_target_pair, domains))
        sampled_data = get_random_sample(data, sample_size, "uniform", self.seed)
        self._samples, self._domains = zip(*sampled_data)

        self.classes = ["background", "wheat_head"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v, in self.class_to_idx.items()}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def samples(self) -> List[Tuple[str, Dict[str, ndarray]]]:
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_data_from_src(root: Path, split: str) -> Any:
        label_src = root.joinpath(f"competition_{split}.csv")
        assert label_src.exists()

        data_src = pd.read_csv(str(label_src), header=0)
        return data_src

    @staticmethod
    def _validate_input(sample_size: Union[int, float], split: str) -> None:
        """Validate the input arguments.

        Args:
            sample_size: Union[int, float]
                The number of samples to use. If an integer, the exact number of
                samples to use. If a float, the percentage of samples to use.
            split: str
                The split of the dataset to use. One of "train", "val", "test".
        """
        ImageDatasetFolder._validate_input("uniform", sample_size)
        assert split in ["train", "val", "test"], f"{split} split is not supported"

    @staticmethod
    def _load_image(file_path: str):
        """Load the image from the file path.

        Args:
            file_path: str
                The path to the image file.

        Returns:
            The image as a numpy array.
        """
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def make_dataset(
        root: Path, data_src: pd.DataFrame
    ) -> List[Tuple[str, Dict[str, ndarray]]]:
        """Create the a list of path-target pairs, representing the dataset.

        Args:
            root: Path
                The root directory of the dataset.
            data_src: pd.DataFrame
                The data source.

        Returns:
            The dataset as a list of tuples of the form (image_path, target).
        """
        filenames = data_src["image_name"].values
        boxes = [GWHD2021.decodeString(item) for item in data_src["BoxesString"].values]
        assert len(filenames) == len(boxes)

        path_target_pair = []
        for idx in range(len(filenames)):
            bboxes = np.asarray(boxes[idx])
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

            num_objs = len(bboxes)
            labels = np.ones((num_objs,), dtype=np.int64)  # there is only one class
            iscrowd = np.zeros(
                (num_objs,), dtype=np.int64
            )  # suppose all instances are not crowd

            target = {
                "boxes": bboxes,
                "labels": labels,
                "area": area,
                "iscrowd": iscrowd,
            }

            file_path = root.joinpath("images", filenames[idx])
            assert file_path.exists()

            path_target_pair.append((str(file_path), target))

        return path_target_pair

    @staticmethod
    def _convert_all_to_tensors(target: Dict[str, ndarray]) -> Dict[str, Tensor]:
        """Convert all the values in the target dictionary to tensors.

        Args:
            target: Dict[str, ndarray]
                The target dictionary.

        Returns:
            The target dictionary with all the values converted to tensors.
        """
        GWHD2021._check_target(target)

        tensor_target = {}
        if len(target["boxes"]) > 0:
            for k in target.keys():
                if k == "boxes":
                    tensor_target[k] = torch.stack(
                        [torch.tensor(item, dtype=torch.float32) for item in target[k]]
                    )
                else:
                    tensor_target[k] = torch.as_tensor(target[k])
        else:
            for k in target.keys():
                if k == "boxes":
                    tensor_target[k] = torch.zeros((0, 4), dtype=torch.float32)
                else:
                    tensor_target[k] = torch.zeros(0, dtype=torch.int64)

        return tensor_target

    @staticmethod
    def _check_target(target: Dict[str, Any]):
        """Check that the target dictionary contains the neccessary items.

        Args:
            target: Dict[str, Any]
                The target dictionary.
        """
        assert (
            set(["boxes", "labels", "area", "iscrowd"]) - set(list(target.keys()))
            == set()
        )

    def __getitem__(self, index: int) -> Tuple[ndarray, Dict]:
        """Get the image and target at the given index."""
        file_path, target = self.samples[index]
        self._check_target(target)

        img = self._load_image(file_path)

        if self.transform:
            transformed = self.transform(
                image=img,
                bboxes=target["boxes"],
                labels=target["labels"],
                area=target["area"],
                iscrowd=target["iscrowd"],
            )
            img = transformed["image"]
            target["boxes"] = transformed["bboxes"]
            target["labels"] = transformed["labels"]
            target["area"] = transformed["area"]
            target["iscrowd"] = transformed["iscrowd"]

        # convert everything to torch.Tensor
        target = self._convert_all_to_tensors(target)
        target["image_id"] = torch.tensor([index])  # coco api requirement
        target["domain"] = torch.tensor([self._domains[index]])

        return img, target

    @staticmethod
    def decodeString(box_str: str):
        """
        Small method to decode the BoxesString
        """
        if box_str == "no_box":
            return np.zeros((0, 4))
        else:
            try:
                boxes = []
                for box in box_str.split(";"):
                    coords = box.split(" ")
                    xmin = int(coords[0])
                    ymin = int(coords[1])
                    xmax = int(coords[2])
                    ymax = int(coords[3])
                    boxes.append(np.array([xmin, ymin, xmax, ymax]))
                return np.array(boxes)
            except:
                print("BoxString is not well formatted. Empty boxes will be returned")
                return np.zeros((0, 4))
