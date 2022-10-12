import os
from typing import Dict, List

import numpy as np

import pytorch_tao as tao
from ignite.engine import Events

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from pytorch_tao.trackers.base import Tracker
from pytorch_tao.plugins import TrainPlugin


class WandbTracker(Tracker, TrainPlugin):
    @tao.ensure_config("log_dir", "wandb_project")
    def __init__(self, name):
        super().__init__()
        if wandb is None:
            raise ModuleNotFoundError("install wandb to use WandbTracker")
        if hasattr(tao.cfg, "wandb_api_key") and tao.cfg.wandb_api_key is not None:
            os.environ["WANDB_API_KEY"] = tao.cfg.wandb_api_key
        self.wandb = wandb.init(
            dir=tao.cfg.log_dir,
            project=tao.cfg.wandb_project,
            name=name,
            group=os.environ.get("TAO_TUNE")
        )

    def add_image(self, image_name: str, images: List[np.ndarray]):
        if not isinstance(images, list):
            images = [images]
        wandb.log({image_name: wandb.Image(images)})

    def add_points(self, points: Dict):
        wandb.log(points)

    @tao.on(Events.COMPLETED)
    def clean(self):
        self.wandb.finish()
