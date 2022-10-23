"""trackers is used"""

import pytorch_tao as tao
from pytorch_tao.trackers.base import Tracker
from pytorch_tao.trackers.wandb_tracker import WandbTracker


def set_tracker(tracker: Tracker):
    """Globally set tracker.

    Should be called after argument parsed so that some trackers will
    record the argument passed in.
    """
    tao.tracker = tracker


__all__ = ["Tracker", "WandbTracker", "set_tracker"]
