import numpy as np
import torch
from pytorch_tao import helper


def test_is_scalar():
    assert helper.is_scalar(1)
    assert helper.is_scalar(1.0)
    assert helper.is_scalar(torch.tensor(1.0))
    assert helper.is_scalar(np.array(1.0))

    assert not helper.is_scalar("123")
    assert not helper.is_scalar(torch.rand((2, 2)))
    assert not helper.is_scalar(np.random.rand(2, 2))
