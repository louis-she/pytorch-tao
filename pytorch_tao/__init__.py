__version__ = "0.1.9"

import logging
from pathlib import Path
from typing import Any

from optuna import Study, Trial

from pytorch_tao.args import _ArgSet, arg, arguments

from pytorch_tao.core import (
    ArgMissingError,
    ConfigMissingError,
    ensure_arg,
    ensure_config,
    init_from_env,
    load_cfg,
    on,
)
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
log_dir: Path = None
name: str = None
tune: bool = False

init_from_env()


__all__ = [
    "Repo",
    "load_cfg",
    "cfg",
    "ensure_config",
    "ensure_arg",
    "ConfigMissingError",
    "ArgMissingError",
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
