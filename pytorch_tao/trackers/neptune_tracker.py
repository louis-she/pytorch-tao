import tempfile
from typing import Dict, List

import ignite.distributed as idist
import matplotlib.pyplot as plt

import numpy as np
from ignite.engine import Events

import pytorch_tao as tao

try:
    import neptune.new as neptune
except ModuleNotFoundError:
    neptune = None

try:
    import plotly
except ModuleNotFoundError:
    plotly = None

from pytorch_tao.trackers.base import Tracker


class NeptuneTracker(Tracker):
    """Tracker with neptune"""

    @idist.one_rank_only()
    def __init__(self, name):
        super().__init__()
        self.logger = tao.get_logger(__name__)
        if neptune is None:
            raise ModuleNotFoundError("install neptune to use NeptuneTracker")
        if plotly is None:
            self.logger.warn(
                "to get best experience with netptune, considering install plotly"
            )
        self.name = name
        self.init()

    @idist.one_rank_only()
    @tao.ensure_config("neptune_project")
    def init(self):
        self.run = neptune.init_run(
            project=tao.cfg.neptune_project,
            name=self.name,
            api_token=tao.cfg.neptune_api_key,
        )

    @idist.one_rank_only()
    def add_image(self, image_name: str, images: List[np.ndarray]):
        if not isinstance(images, list):
            images = [images]
        # TODO: we can have more control with the figsize depending
        #       on the shapes of the images
        fig, axes = plt.subplots(1, len(images), figsize=(10, 10))
        for ax, image in zip(axes, images):
            ax.imshow(image)
        self.run[f"images/{image_name}"].log(neptune.types.File.as_image(fig))

    @idist.one_rank_only()
    def add_histogram(self, name, data: List[float], bins=64):
        fig, ax = plt.subplots()
        ax.hist(data, bins=bins)
        self.run[f"histogram/{name}"].log(neptune.types.File.as_image(fig))

    @idist.one_rank_only()
    def add_points(self, points: Dict):
        for key, value in points.items():
            self.run[f"charts/{key}"].append(value)

    @idist.one_rank_only()
    def update_meta(self, meta: dict):
        for key, value in meta.items():
            self.run[f"meta/{key}"] = value

    @idist.one_rank_only()
    def add_tabular(self, name, df):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w") as fp:
            df.to_csv(fp.name, index=False)
            self.run[f"tabular/{name}"].upload(fp.name, wait=True)

    @idist.one_rank_only()
    @tao.on(Events.COMPLETED)
    def clean(self):
        self.run.stop()
