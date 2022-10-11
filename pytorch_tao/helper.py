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
