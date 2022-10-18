import warnings

import torch
from ignite.engine import Events

import pytorch_tao as tao
from pytorch_tao.plugins import base


class Scheduler(base.TrainPlugin):
    def __init__(self, torch_scheduler: torch.optim.lr_scheduler._LRScheduler):
        super().__init__()
        self._scheduler = torch_scheduler

    @tao.on(Events.ITERATION_COMPLETED)
    def _step(self):
        with warnings.catch_warnings(record=True):
            # ommit the annoying step order warning...
            self._scheduler.step()
        tao.tracker.add_points({"lr": self._scheduler.get_last_lr()[0]})
