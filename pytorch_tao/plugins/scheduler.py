import torch
from pytorch_tao.plugins import base, events
import pytorch_tao as tao


class Scheduler(base.TrainPlugin):

    def __init__(self, torch_scheduler: torch.optim.lr_scheduler._LRScheduler):
        super().__init__()
        self._scheduler = torch_scheduler

    @events.iteration_completed()
    def _step(self):
        pass
