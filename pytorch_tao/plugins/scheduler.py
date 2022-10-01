import torch
from pytorch_tao.plugins import base
import pytorch_tao as tao
from ignite.engine import Events


class Scheduler(base.TrainPlugin):

    def __init__(self, torch_scheduler: torch.optim.lr_scheduler._LRScheduler):
        super().__init__()
        self._scheduler = torch_scheduler

    @tao.on(Events.ITERATION_COMPLETED)
    def _step(self):
        self._scheduler.step()
        tao.tracker.add_points({"lr": self._scheduler.get_lr()[0]})
