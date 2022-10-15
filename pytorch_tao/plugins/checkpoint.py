from typing import Dict

from ignite.engine import Events
from ignite.handlers import Checkpoint as ICheckpoint, DiskSaver

import pytorch_tao as tao
from pytorch_tao.plugins.base import ValPlugin


class Checkpoint(ValPlugin):
    def __init__(self, metric_name: str, objects: Dict, score_sign: int = 1, n_saved=3):
        self.metric_name = metric_name
        self.objects = objects
        self.save_path = tao.log_dir / "checkpoints"
        self._checkpoint = ICheckpoint(
            self.objects,
            DiskSaver(
                self.save_path.as_posix(),
                create_dir=True,
                save_on_rank=0,
                require_empty=False,
            ),
            score_function=ICheckpoint.get_default_score_fn(metric_name, score_sign),
            n_saved=n_saved,
        )

    @tao.on(Events.EPOCH_COMPLETED)
    def _save_checkpoint(self):
        self._checkpoint(self.engine)
