import sys

sys.path.append("..")
from utils.registry import Registry

MODELS = Registry("models")
BACKBONES = Registry("backbones")
HEADS = Registry("heads")
