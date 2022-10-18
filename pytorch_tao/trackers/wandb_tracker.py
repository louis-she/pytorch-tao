import os
from typing import Dict, List

import numpy as np
from ignite.engine import Events

import pytorch_tao as tao

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from pytorch_tao.plugins import TrainPlugin
from pytorch_tao.trackers.base import Tracker


class WandbTracker(Tracker, TrainPlugin):
    @tao.ensure_config("wandb_project")
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
            group=os.environ.get("TAO_TUNE"),
        )
        if tao.args:
            self.update_meta(tao.args.dict())

    def add_image(self, image_name: str, images: List[np.ndarray]):
        if not isinstance(images, list):
            images = [images]
        images = [wandb.Image(image) for image in images]
        wandb.log({image_name: images})

    def add_points(self, points: Dict):
        wandb.log(points)

    def update_meta(self, meta: dict):
        wandb.config.update(meta)

    @tao.on(Events.COMPLETED)
    def clean(self):
        self.wandb.finish()
