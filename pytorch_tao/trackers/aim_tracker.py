from typing import Dict, List

import ignite.distributed as idist

import numpy as np
from ignite.engine import Events

import pytorch_tao as tao

try:
    import aim
except ModuleNotFoundError:
    aim = None

from pytorch_tao.trackers.base import Tracker


class AimTracker(Tracker):
    """Tracker with aim"""

    @idist.one_rank_only()
    def __init__(self):
        super().__init__()
        if aim is None:
            raise ModuleNotFoundError("install aim to use AimTracker")
        self.init()

    @idist.one_rank_only()
    @tao.ensure_config("aim_repo")
    def init(self):
        self.run = aim.Run(
            repo=tao.cfg.aim_repo,
            experiment=tao.cfg.aim_experiment_name,
            log_system_params=True,
        )

    @idist.one_rank_only()
    def add_image(self, image_name: str, images: List[np.ndarray]):
        if not isinstance(images, list):
            images = [images]
        for image in images:
            image = aim.Image(image)
            self.run.track(image, name=image_name, step=self.trainer.state.iteration)

    @idist.one_rank_only()
    def add_histogram(self, name, data: List[float], bins=64):
        self.run.track(
            aim.Distribution(samples=data, bin_count=bins),
            name=name,
            step=self.trainer.state.iteration,
        )

    @idist.one_rank_only()
    def add_points(self, points: Dict):
        self.run.track(points, step=self.trainer.state.iteration)

    @idist.one_rank_only()
    def update_meta(self, meta: dict):
        for key, value in meta.items():
            self.run[key] = value

    @idist.one_rank_only()
    @tao.on(Events.COMPLETED)
    def clean(self):
        self.run.close()
