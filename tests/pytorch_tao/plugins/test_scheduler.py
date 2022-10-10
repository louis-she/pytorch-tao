import pytest
import pytorch_tao as tao
import torch
from pytorch_tao.plugins.scheduler import Scheduler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR


@pytest.fixture(scope="function")
def optimizer():
    return SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)


@pytest.fixture(scope="function")
def func_counter():
    def func_counter(func):
        def _real(*args, **kwargs):
            _real.called_count += 1
            return func(*args, **kwargs)

        _real.called_count = 0
        return _real

    return func_counter


def test_scheduler_step_lr(trainer: tao.Trainer, optimizer: SGD, func_counter):
    step_scheduler = StepLR(optimizer, step_size=100)
    step_scheduler.step = func_counter(step_scheduler.step)
    _ori_add_points = tao.tracker.add_points
    tao.tracker.add_points = func_counter(tao.tracker.add_points)
    trainer.use(Scheduler(step_scheduler))
    trainer.fit(max_epochs=5)
    assert step_scheduler.step.called_count == 500
    assert tao.tracker.add_points.called_count == 500
    tao.tracker.add_points = _ori_add_points
