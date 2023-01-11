import os
import sys

from typing import Dict

import numpy as np
from ignite.engine import Events

import pytorch_tao as tao
from pytorch_tao.plugins import TrainPlugin


class Tracker(TrainPlugin):
    """Base class for all other trackers"""

    @tao.on(Events.STARTED)
    def _tracker_reproduce_command(self):
        if tao.repo.is_dirty():
            return
        self.update_meta(
            {
                "reproduce_command": f"tao run --checkout {tao.repo.head_hexsha(True)} "
                f"{sys.argv[0]} {tao.args.get_command()}",
            }
        )

    @tao.on(Events.STARTED)
    def _tracker_tao_environment(self):
        tao_env = {
            key: value for key, value in os.environ.items() if key.startswith("TAO_")
        }
        self.update_meta(tao_env)

    @tao.on(Events.STARTED)
    def _tracker_parameters(self):
        self.update_meta(tao.args.dict())

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
