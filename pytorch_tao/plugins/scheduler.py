import warnings

import torch
from ignite.engine import Events

import pytorch_tao as tao
from pytorch_tao.plugins import base


class Scheduler(base.TrainPlugin):
    """Simple adapter for PyTorch schedulers.

    Args:
        torch_scheduler: pytorch built-in scheduler.

    ..  list-table:: Hooks
        :header-rows: 1

        * - Hook Point
          - Logic
        * - ITERATION_COMPLETED
          - call `scheduler.step` and track the learning rate

    .. code-block:: python

        import pytorch_tao as tao
        from pytorch_tao.plugin import Scheduler
        from torch.optim.lr_scheduler import StepLR

        model = ...
        optimizer = ...

        trainer = tao.Trainer()
        trainer.use(Scheduler(StepLR(optimizer, step_size=30)))
    """

    def __init__(self, torch_scheduler: torch.optim.lr_scheduler._LRScheduler):
        super().__init__()
        self._scheduler = torch_scheduler

    @tao.on(Events.ITERATION_COMPLETED)
    def _step(self):
        try:
            with warnings.catch_warnings(record=True):
                # ommit the annoying step order warning...
                self._scheduler.step()
        except ValueError:
            pass
        tao.tracker.add_points({"lr": self._scheduler.get_last_lr()[0]})
