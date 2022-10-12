from typing import Dict
import pytorch_tao as tao

import numpy as np
try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from pytorch_tao.trackers.base import Tracker


class WandbTracker(Tracker):

    @tao.ensure_config("log_path", "wandb_project")
    def __init__(self):
        self.wandb = wandb.init(
            dir=tao.cfg.log_path,
            project=tao.cfg.wandb_project,
            name=tao.args.name
        )

    def add_image(self, image: np.ndarray):
        super().add_image(image)

    def add_points(self, points: Dict):
        super().add_points(points)
