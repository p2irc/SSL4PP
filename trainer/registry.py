import sys
sys.path.append("..")
from utils.registry import Registry

OPTIMIZERS = Registry("optimizers")
LR_SCHEDULERS = Registry("lr_schedulers")
LOSSES = Registry("losses")