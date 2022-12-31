import random

import numpy as np
import pytest
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


def test_seedeverything():
    helper.seed_everything(777)
    random_tensor = torch.rand([2, 2])
    random_ndarray = np.random.rand(2, 2)
    random_float = random.random()
    helper.seed_everything(777)
    assert torch.rand([2, 2]).sum().item() == random_tensor.sum().item()
    assert np.random.rand(2, 2).sum() == random_ndarray.sum()
    assert random.random() == random_float


def test_seedeverything_range():
    with pytest.raises(ValueError, match="seed number should be less than 10000"):
        helper.seed_everything(10000)
    helper.seed_everything(9999)


def test_stride_event_filter():
    func = helper.stride_event_filter()
    for i in range(1, 10):
        assert func(None, i)

    func = helper.stride_event_filter("::")
    for i in range(1, 10):
        assert func(None, i)

    func = helper.stride_event_filter("10:3")
    for e in [10, 13, 16, 19, 310]:
        assert func(None, e)
    for e in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18]:
        assert not func(None, e)

    func = helper.stride_event_filter("10:3:20")
    for e in [10, 13, 16, 19]:
        assert func(None, e)

    func = helper.stride_event_filter("4:2:8")
    for e in [4, 6, 8]:
        assert func(None, e)
    for e in [1, 2, 3, 5, 7, 9]:
        assert not func(None, e)
