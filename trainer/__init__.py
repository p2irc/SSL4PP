from collections import Counter
import inspect
from typing import List, Optional
import warnings
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from trainer.registry import OPTIMIZERS, LR_SCHEDULERS
from .losses import *
from .metrics import *

optim_classes = inspect.getmembers(optim, inspect.isclass)
for cls in optim_classes:
    if cls[0] != 'Optimizer':
        OPTIMIZERS.register_class(cls[1])

schedulers = inspect.getmembers(lr_scheduler, inspect.isclass)
for cls in schedulers:
    if "LR" in cls[0] or "CosineAnnealing" in cls[0] and cls[0] != 'LambdaLR':
        LR_SCHEDULERS.register_class(cls[1])

#@LR_SCHEDULERS.register_class
class ConstantLR(LambdaLR):
    def __init__(self, optimizer, verbose=False) -> None:
        lr_lambda = lambda epoch: (epoch+1) // (epoch+1)
        last_epoch = -1
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)

@LR_SCHEDULERS.register_class
class MultiStepLRWithLinearWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones: List,
        warmup_steps: float,
        warmup_ratio: Optional[float] = 0.001,
        gamma: Optional[float] = 0.1,
        last_epoch: Optional[int] = -1,
        verbose: Optional[bool] = False) -> None:

        self.milestones = Counter(milestones)
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch <= self.warmup_steps:
            return [(self.warmup_ratio * group['initial_lr']) + 
            ((group['initial_lr'] - (self.warmup_ratio * group['initial_lr'])) / self.warmup_steps) * self.last_epoch
                        for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]