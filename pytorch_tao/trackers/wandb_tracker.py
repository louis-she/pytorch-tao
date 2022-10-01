from typing import Dict

import numpy as np

from pytorch_tao.trackers.base import Tracker


class WandbTracker(Tracker):
    def __init__(self):
        pass

    def add_image(self, image: np.ndarray):
        super().add_image(image)

    def add_points(self, points: Dict):
        super().add_points(points)
