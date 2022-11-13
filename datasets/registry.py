import sys

sys.path.append("..")
from utils.registry import Registry

DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
