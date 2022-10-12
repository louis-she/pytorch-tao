__version__ = "0.1.6"

import logging
import os
from typing import Any

from optuna import load_study, Study, Trial

from pytorch_tao.args import _ArgSet, arg, arguments

from pytorch_tao.core import ConfigMissingError, ensure_config, ensure_arg, load_cfg, on
from pytorch_tao.repo import DirtyRepoError, Repo
from pytorch_tao.trackers import set_tracker, Tracker
from pytorch_tao.trainer import Trainer
from pytorch_tao.tune import tell

logging.captureWarnings(True)


args: _ArgSet = None
study: Study = None
trial: Trial = None
cfg: Any = None
repo: Repo = None
tracker = Tracker()

if os.getenv("TAO_REPO"):
    repo = Repo(os.getenv("TAO_REPO"))

if os.getenv("TAO_TUNE"):
    study = load_study(study_name=os.getenv("TAO_TUNE"), storage=cfg.study_storage)
    # trial should been asked after the argument parsing phase


__all__ = [
    "Repo",
    "load_cfg",
    "cfg",
    "ensure_config",
    "ensure_arg",
    "ConfigMissingError",
    "args",
    "study",
    "trial",
    "DirtyRepoError",
    "arguments",
    "arg",
    "tell",
    "on",
    "Trainer",
    "set_tracker",
]
