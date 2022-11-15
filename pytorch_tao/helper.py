import random
from typing import Any

import numpy as np
import torch


def is_scalar(scalar: Any):
    if isinstance(scalar, (float, int)):
        return True
    if isinstance(scalar, (torch.Tensor, np.ndarray)) and len(scalar.shape) == 0:
        return True
    return False


def item(scalar: Any):
    if isinstance(scalar, (float, int)):
        return scalar
    if isinstance(scalar, (torch.Tensor, np.ndarray)):
        return scalar.item()


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
