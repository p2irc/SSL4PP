"""Transforms for self-supervised learning tasks."""
from copy import deepcopy
from typing import Any, List

from albumentations.core.composition import BaseCompose
from datasets.registry import TRANSFORMS


@TRANSFORMS.register_class
class TwoCropsTransform(BaseCompose):
    """Take two random crops of one image as the query and key.

    https://github.com/facebookresearch/simsiam/blob/main/simsiam/loader.py

    """

    def __init__(self, base_transform):
        """Initialize the transform."""
        super().__init__(transforms=base_transform, p=1.0)

    def __call__(self, force_apply=False, **data) -> List[Any]:
        """Apply the transform to the data."""
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        query = deepcopy(data)
        key = deepcopy(data)
        if self.transforms:
            for t in self.transforms:
                query = t(force_apply=force_apply, **query)
                key = t(force_apply=force_apply, **key)
        return [query, key]
