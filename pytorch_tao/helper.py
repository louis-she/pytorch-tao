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


def stride_event_filter(event: str = None):
    """Create a stride event filter

    Args:
        event: the pattern of the stride. pattern is start:stride:end.
            By default, start = 1, stride = 1, end = -1, where -1 means infinate.
            Note that the start and end are both inclusive.

        examples:
            None or "" or "1" or "1:1" or "1:1:-1 means every stride.
            "1:1:10" means starts from 1, stride is 1, and ends at 10(not include 10).
            "1:1:1" means only the first event.
            "1:1:0" is stil valid, but is empty.
            "2" or "2:1" or "2:1:-1 means stride 1 start from 2.
    """
    if event is None:
        event = "1:1:-1"
    event = [x for x in event.split(":")]
    start = int(event[0]) if len(event) > 0 and event[0] != "" else 1
    stride = int(event[1]) if len(event) > 1 and event[1] != "" else 1
    end = int(event[2]) if len(event) > 2 and event[2] != "" else -1
    if end == -1:
        end = float("inf")

    def _event_filter(_, e):
        return (e >= start) and (e <= end) and ((e - start) % stride == 0)

    return _event_filter
