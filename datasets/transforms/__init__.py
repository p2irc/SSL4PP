import inspect
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.registry import TRANSFORMS
from .ssl_transforms import TwoCropsTransform

TRANSFORMS.register_class(ToTensorV2)

albu_aug_classes = inspect.getmembers(A, inspect.isclass)
for cls in albu_aug_classes:
    TRANSFORMS.register_class(cls[1])
