from typing import Dict
from pytorch_tao.trackers.base import Tracker
import numpy as np


class WandbTracker(Tracker):

    def __init__(self):
        pass

    def add_image(self, image: np.ndarray):
        super().add_image(image)

    def add_points(self, points: Dict):
        super().add_points(points)
