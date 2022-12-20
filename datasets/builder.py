"""Factory for building datasets and transforms."""
import random
import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utils.distributed as dist_utils
from datasets.registry import DATASETS, TRANSFORMS
from utils.registry import build_from_cfg

from .sampler import DistributedEvalSampler
from .transforms import TwoCropsTransform


def build_dataset(
    cfg: DictConfig, task_type: str, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Build training and test datasets.

    Args:
        cfg: DictConfig)
            A hydra config object containing all the information needed to build
            the dataset
        task_type: str
            The type of task the dataset is used for.
        seed: int
            The random seed for random sampler.

    Return:
        A dictionary with two keys, train and test, representing the training
        and test datasets.

    """
    # modify the config object
    args = deepcopy(cfg)
    common_cfg = {_key: _value for _key, _value in args.items() if "split" not in _key}
    common_cfg["seed"] = seed

    # enforce the return dictionary to follow a standard format
    # {"train": train/train_mini_dataset, "test": val/test_dataset}
    dataset_dict = {"train": None, "test": None}

    # build the train dataset
    train_cfg = {**common_cfg, **args["train_split"]}
    train_cfg["transform"] = _compose_transforms(
        build_transforms(train_cfg["transform"]), task_type
    )
    dataset_dict["train"] = build_from_cfg(train_cfg, DATASETS)

    # build the test dataset
    if args.get("test_split"):
        test_cfg = {**common_cfg, **args["test_split"]}
        if test_cfg.get("transform"):
            test_cfg["transform"] = _compose_transforms(
                build_transforms(test_cfg["transform"]), task_type
            )
        dataset_dict["test"] = build_from_cfg(test_cfg, DATASETS)

    return dataset_dict


def build_transforms(cfg: Union[ListConfig, List]) -> List:
    """Build an Albumentations data augmentation pipeline.

    Args:
        cfg: Union[ListConfig, List]:
            A hydra config object or a list containing all the information
            needed to build the augmentation pipeline.

    Return:
        A list of Alubmentations objects

    """
    if cfg is None or not isinstance(cfg, (list, ListConfig)):
        warnings.warn(
            f"Expected a list or ListConfig as input, got {type(cfg)}. \
             The program will continue.",
            UserWarning,
        )
        return None

    aug_list = []
    for transform in cfg:
        if "transforms" in transform.keys():
            component_list = build_transforms(transform["transforms"])

            if component_list:
                transform = dict(transform)
                transform["transforms"] = component_list
                aug_list.append(build_from_cfg(transform, TRANSFORMS))
        else:
            aug_list.append(build_from_cfg(transform, TRANSFORMS))

    return aug_list


def _compose_transforms(transform_list: List, task_type: str) -> A.Compose:
    """Composes an Albumentations pipeline based on the type of task.

    Args:
        transform_list: List
            A list of Albumentations classes.
        task_type: str
            The type of task the transform pipeline is for.

    Return:
        A compose object.

    """
    if task_type in ["SimSiam", "ContrastiveLearning"]:
        transforms = A.Compose([TwoCropsTransform(transform_list)])
    elif task_type == "ObjectDetection":
        transforms = A.Compose(
            transform_list,
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["labels", "area", "iscrowd"]
            ),
        )
    elif task_type == "DensityCounting":
        transforms = A.Compose(
            transform_list,
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["class_labels"]
            ),
        )
    else:
        transforms = A.Compose(transform_list)
    return transforms


def load_datasets(
    datasets: Dict,
    batch_size: int,
    num_workers: int,
    drop_last: Optional[bool] = True,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    """Initialize the PyTorch Dataloader object for the given dataset(s).

    Args:
        datasets: Dict
            A dictionary containing the train and test datasets.
        batch_size: int
            The number of samples per batch to be loaded.
        num_workers: int
            The number of processors to be used for data loading.
        seed: int
            The random seed for deterministic data loading.
        drop_last: bool
            Whether to drop the last incomplete batch for the training dataset.
        kwargs: Any
            Additional keyword arguments to be passed to the DataLoader object.
            Example: collate_fn -> a custom function for creating batches.

    Return:
        A tuple of DataLoader objects, for the training and test datasets.

    """
    train_dataset = datasets["train"]
    distributed = dist_utils.is_dist_avail_and_initialized()

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, seed=seed, drop_last=True)

    init_fn = (
        partial(
            worker_init_fn,
            num_workers=num_workers,
            rank=dist_utils.get_rank(),
            seed=seed,
        )
        if seed is not None
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        pin_memory=True,
        drop_last=drop_last,
        batch_size=batch_size,
        sampler=train_sampler,
        worker_init_fn=init_fn,
        num_workers=num_workers,
        shuffle=(train_sampler is None),
        collate_fn=kwargs.get("collate_fn"),
    )

    test_dataset = datasets["test"]
    test_loader = None
    if test_dataset is not None:
        test_sampler = None
        if distributed:
            test_sampler = DistributedEvalSampler(test_dataset, shuffle=False)

        test_loader = DataLoader(
            test_dataset,
            batch_size=kwargs.get("test_batch_size", batch_size),
            num_workers=num_workers,
            sampler=test_sampler,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=init_fn,
            collate_fn=kwargs.get("collate_fn"),
        )

    return train_loader, test_loader


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int) -> None:
    """A function to initilaze dataloader workers.

    Args:
        worker_id: int
            ID of the worker process.
        num_workers: int
            Total number of workers that will be initialized.
        rank: int
            The rank of the current process.
        seed: int
            A random seed used determine the worker seed.

    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
