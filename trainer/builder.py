from copy import deepcopy
from typing import Any, Iterable, Optional

import torch.optim as optim
from omegaconf import DictConfig

from utils.registry import build_from_cfg
from trainer.registry import LOSSES, LR_SCHEDULERS, OPTIMIZERS

def build_optimizer(
    cfg: DictConfig,
    parameters: Iterable,
    init_lr: float,
    use_lars: Optional[bool]=False) -> optim.Optimizer:
    """
    Configures the optimizer as specified in the cfg.

    Args:
        cfg (DictConfig): a hydra config object containing all the information
            needed to build the optimizer.
        parameters (Iterable): a list of model parameters to be optimized.
        init_lr (float): initial learning rate.
        use_lars (bool): whether to wrap the optimizer with the LARS optimizer.

    Return:
        The initialized optimizer object.
    """
    args = deepcopy(dict(cfg))
    args['params'] = parameters
    args['lr'] = init_lr

    optimizer = build_from_cfg(args, OPTIMIZERS)

    if use_lars:
        from apex.parallel.LARC import LARC
        print("Using LARS optimizer")
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)
    
    return optimizer

def build_scheduler(
        cfg: DictConfig,
        optimizer: optim.Optimizer,
        init_lr: float,
        with_lars: bool,
        **kwargs: Any) -> optim.lr_scheduler:
    """Constructs a learning rate scheduler object based on the given cfg.

    Args:
        cfg (DictConfig):a hydra config object.
        optimizer (optim.Optimizer): the optimizer for the scheduler to be based on.
        init_lr (float): intial learning rate.
        with_lars (bool): whether the LARS optimizer is being used.
    
    Return:
        The initialized learning rate scheduler.
    """
    args = deepcopy(dict(cfg))
    if with_lars:
        optimizer = optimizer.optim

    args['optimizer'] = optimizer
    if args['type'] == 'OneCycleLR':
        args['max_lr'] = init_lr
        args['total_steps'] = kwargs['total_steps']
    if args['type'] == 'CosineAnnealingLR':
        args['T_max'] = kwargs['T_max']

    schedule = build_from_cfg(args, LR_SCHEDULERS)
    return schedule

def build_loss(cfg: DictConfig):
    """
    Constructs the loss function specified in cfg.

    Args:
        cfg (DictConfig): a hydra config object
    """
    args = deepcopy(dict(cfg))
    criterion = build_from_cfg(args, LOSSES)

    return criterion
