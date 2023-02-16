from typing import Dict
import os

import ignite.distributed as idist

from ignite.engine import Events
from ignite.handlers import Checkpoint as ICheckpoint, DiskSaver, global_step_from_engine

import pytorch_tao as tao
from pytorch_tao.plugins.base import ValPlugin


class Checkpoint(ValPlugin):
    """Save models or any other states.
    This is just a wrapper of :class:`ignite:ignite.handlers.checkpoint.Checkpoint`.

    Args:
        metric_name: the name of the metric to determine which is the best model to save.
        objects: objects to save.
        score_sign: 1 higher the better or -1 lower the better.
        n_saved: how many file to keep.

    ..  list-table:: Hooks
        :header-rows: 1

        * - Hook Point
          - Logic
        * - EPOCH_COMPLETED
          - saving the states

    .. code-block:: python

        import pytorch_tao as tao

        model = ...
        trainer = tao.Trainer()

        # save the top-3 models of accuracy
        trainer.use(Checkpoint("accuracy", {"model": model}))
    """

    @idist.one_rank_only()
    def __init__(self, metric_name: str, objects: Dict, score_sign: int = 1, n_saved=3):
        self.metric_name = metric_name
        self.objects = objects
        self.score_sign = score_sign
        self.n_saved = n_saved
        self.save_path = tao.log_dir / os.environ["TAO_ID"] / "checkpoints"

    @idist.one_rank_only()
    def after_use(self):
        self._checkpoint = ICheckpoint(
            self.objects,
            DiskSaver(
                self.save_path.as_posix(),
                create_dir=True,
                save_on_rank=0,
                require_empty=False,
            ),
            score_function=ICheckpoint.get_default_score_fn(self.metric_name, self.score_sign),
            n_saved=self.n_saved,
            global_step_transform=global_step_from_engine(self.trainer.train_engine),
        )

    @idist.one_rank_only()
    @tao.on(Events.EPOCH_COMPLETED)
    def _save_checkpoint(self):
        self._checkpoint(self.engine)
