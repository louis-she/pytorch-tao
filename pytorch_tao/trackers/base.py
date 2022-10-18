from typing import Dict

import numpy as np


class Tracker:
    def set_global_step(self, step: int):
        self.global_step = step

    def add_points(self, points: Dict):
        pass

    def add_image(self, image: np.ndarray):
        pass

    def update_meta(self, meta: dict):
        pass
