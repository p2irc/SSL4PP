"""The Open Plant Phenotyping Database (OPPD)."""
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from numpy import ndarray

from datasets.gwhd_2021 import GWHD2021
from datasets.registry import DATASETS

from .file_src_dataset import FileSrcDataset


@DATASETS.register_class
class OPPDFull(FileSrcDataset):
    """The Open Plant Phenotyping Database (OPPD).

    Args:
        root: str
            The root directory of the dataset.
        split: Optional[str]
            The split of the dataset to use. One of "train", "test".
        subset: Optional[int]
            The subset of the dataset to use. One of 0, 1, 2.
        crop_out_black_pixels: Optional[bool]
            Whether to crop out black pixels from the image.
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
        crop_out_black_pixels: Optional[bool] = True,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Init method."""
        super().__init__(root, transform, seed)

        src_path = self.root.joinpath(f"{split}_{subset}.json")
        data_src = self.load_from_json(src_path)

        self._img_ids = [item["image_id"] for item in data_src]
        self._samples = self.make_dataset(self.root, data_src)

        self.classes = ["background", "1PLAK"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v, in self.class_to_idx.items()}

        self.crop_black = crop_out_black_pixels

    @property
    def num_classes(self):
        """The number of classes in the dataset."""
        return len(self.classes)

    @property
    def samples(self) -> List[Tuple[str, Dict[str, ndarray]]]:
        """The samples of the dataset."""
        return self._samples

    def __len__(self):
        """The number of samples in the dataset."""
        return len(self.samples)

    @staticmethod
    def make_dataset(
        root: Path, data_src: List[Dict[Any, Any]]
    ) -> List[Tuple[str, Dict[str, ndarray]]]:
        """Create a list of samples containing path-target pairs.

        Args:
            root: Path
                The root directory of the dataset.
            data_src: List[Dict[Any, Any]]
                The data source to use.

        Returns:
            A list of samples of the dataset.

        """
        path_target_pair = []
        for idx in range(len(data_src)):
            filename = root.joinpath(data_src[idx]["filename"])
            assert filename.exists()

            bboxes = []
            for plant_info in data_src[idx]["plants"]:
                bbox_dict = plant_info["bndbox"]
                xmin = bbox_dict["xmin"]
                ymin = bbox_dict["ymin"]
                xmax = bbox_dict["xmax"]
                ymax = bbox_dict["ymax"]
                bbox = np.array([xmin, ymin, xmax, ymax])
                if np.all(bbox == bbox[0]) or xmin >= xmax or ymin >= ymax:
                    continue
                bboxes.append(bbox)

            if len(bboxes) == 0:
                bboxes = np.zeros((0, 4), dtype=np.float32)

            if not isinstance(bboxes, ndarray):
                bboxes = np.array(bboxes)

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

            path_target_pair.append((str(filename), target))

        return path_target_pair

    @staticmethod
    def _modify_target(target: Dict, crop_points: Tuple[float]):
        """Modify the target to account for the cropping of the image.

        Args:
            target: Dict
                The target to modify.
            crop_points: Tuple[float]
                The points to crop the image to.

        Returns:
            The modified target.

        """
        assert len(crop_points) == 4

        modified_target = deepcopy(target)
        bboxes = target["boxes"]
        if bboxes.size > 0:
            xmin = np.minimum(
                np.maximum(0, bboxes[:, 0] - crop_points[0]), crop_points[2]
            )
            ymin = np.minimum(
                np.maximum(0, bboxes[:, 1] - crop_points[1]), crop_points[3]
            )
            xmax = np.minimum(
                np.maximum(0, bboxes[:, 2] - crop_points[0]), crop_points[2]
            )
            ymax = np.minimum(
                np.maximum(0, bboxes[:, 3] - crop_points[1]), crop_points[3]
            )

            modified_bboxes = np.stack([xmin, ymin, xmax, ymax], axis=-1)

            # delete degenerate boxes
            rows_to_delete = []
            for i in range(modified_bboxes.shape[0]):
                if np.all(modified_bboxes[i] == modified_bboxes[i][0]) or (
                    modified_bboxes[i][0] >= modified_bboxes[i][2]
                    or modified_bboxes[i][1] >= modified_bboxes[i][3]
                ):
                    rows_to_delete.append(i)
            good_bboxes = np.delete(modified_bboxes, rows_to_delete, axis=0)

            num_objs = len(good_bboxes)
            modified_target["boxes"] = good_bboxes
            modified_target["labels"] = np.ones((num_objs,), dtype=np.int64)
            modified_target["iscrowd"] = np.zeros((num_objs,), dtype=np.int64)
            modified_target["area"] = (good_bboxes[:, 3] - good_bboxes[:, 1]) * (
                good_bboxes[:, 2] - good_bboxes[:, 0]
            )

        return modified_target

    @staticmethod
    def _load_image(url: str, crop_black: bool):
        """Load an image from a URL.

        Args:
            url: str
                The URL to load the image from.
            crop_black: bool
                Whether to crop out black pixels.

        Returns:
            The loaded image.

        """
        image = cv2.imread(url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if crop_black:
            # crop black background: https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Find the index of the largest contour:
            # https://stackoverflow.com/questions/16538774/dealing-with-contours-and-bounding-rectangle-in-opencv-2-4-python-2-7
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]

            x, y, w, h = cv2.boundingRect(cnt)
            crop = image[y : y + h, x : x + w]
        else:
            crop = image
            x = 0
            y = 0
            h, w = image.shape[:2]

        return crop, (x, y, w, h)

    def __getitem__(self, index: int) -> Tuple[ndarray, Dict]:
        """Get an item from the dataset at the given index."""
        file_path, target = self.samples[index]
        GWHD2021._check_target(target)

        image, crop_points = self._load_image(file_path, self.crop_black)

        if self.crop_black:
            target = self._modify_target(target, crop_points)

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=target["boxes"],
                labels=target["labels"],
                area=target["area"],
                iscrowd=target["iscrowd"],
            )
            image = transformed["image"]
            target["boxes"] = transformed["bboxes"]
            target["labels"] = transformed["labels"]
            target["area"] = transformed["area"]
            target["iscrowd"] = transformed["iscrowd"]

        # convert everything to torch.Tensor
        target = GWHD2021._convert_all_to_tensors(target)
        target["image_id"] = torch.tensor(self._img_ids[index])  # coco api requirement

        return image, target
