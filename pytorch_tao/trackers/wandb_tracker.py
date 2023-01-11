import os
from typing import Dict, List

import ignite.distributed as idist

import numpy as np
from ignite.engine import Events

import pytorch_tao as tao

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from pytorch_tao.trackers.base import Tracker


class WandbTracker(Tracker):
    """Tracker with wandb"""

    @idist.one_rank_only()
    def __init__(self, name):
        super().__init__()
        if wandb is None:
            raise ModuleNotFoundError("install wandb to use WandbTracker")
        self.name = name
        self.init()

    @idist.one_rank_only()
    @tao.ensure_config("wandb_project")
    def init(self):
        if hasattr(tao.cfg, "wandb_api_key") and tao.cfg.wandb_api_key is not None:
            os.environ["WANDB_API_KEY"] = tao.cfg.wandb_api_key
        self.wandb = wandb.init(
            dir=tao.cfg.log_dir,
            project=tao.cfg.wandb_project,
            name=self.name,
            group=os.environ.get("TAO_TUNE"),
        )

    @idist.one_rank_only()
    def add_image(self, image_name: str, images: List[np.ndarray]):
        if not isinstance(images, list):
            images = [images]
        images = [wandb.Image(image) for image in images]
        wandb.log({image_name: images})

    @idist.one_rank_only()
    def add_histogram(self, name, data: List[float], bins=64):
        wandb.log({name: wandb.Histogram(data, num_bins=bins)})

    @idist.one_rank_only()
    def add_points(self, points: Dict):
        wandb.log(points)

    @idist.one_rank_only()
    def update_meta(self, meta: dict):
        wandb.config.update(meta)

    @idist.one_rank_only()
    @tao.on(Events.COMPLETED)
    def clean(self):
        self.wandb.finish()
