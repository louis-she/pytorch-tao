import sys

from typing import Dict

import numpy as np
from ignite.engine import Events

import pytorch_tao as tao
from pytorch_tao.plugins import TrainPlugin


class Tracker(TrainPlugin):
    """Base class for all other trackers"""

    @tao.on(Events.STARTED)
    def _tracker_parameters(self):
        if tao.repo.is_dirty():
            return
        self.update_meta(
            {
                "reproduce_command": f"tao run --checkout {tao.repo.head_hexsha(True)} "
                f"{sys.argv[0]} {tao.args.get_command()}"
            }
        )

    def set_global_step(self, step: int):
        self.global_step = step

    def add_points(self, points: Dict):
        pass

    def add_image(self, image: np.ndarray):
        pass

    def update_meta(self, meta: dict):
        pass

    def add_tabular(self, name, df):
        pass
