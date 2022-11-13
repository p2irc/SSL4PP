import os
import random
import warnings
from typing import Optional

import numpy as np
import torch


def set_random_seed(seed: int, determinstic: Optional[bool] = False) -> None:
    """Globally sets a random seed for python, numpy and PyTorch random
    modules.

    Args:
        seed (int): value of random seed to set
        deterministic (bool): turn on CuDNN deterministic settings. Default: False

    """
    if seed is not None and isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if determinstic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = (
                False  # stops algorithm tuning; slows training
            )
            warnings.warn(
                "You have chosen determinstic training. "
                "This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints."
            )
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
