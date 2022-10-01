import pytorch_tao as tao
from pytorch_tao.trackers.base import Tracker
from pytorch_tao.trackers.wandb_tracker import WandbTracker


def set_tracker(tracker: Tracker):
    tao.tracker = tracker


__all__ = ["Tracker", "WandbTracker", "set_tracker"]
