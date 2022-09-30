__version__ = "0.1.5"

import os
from typing import Any

from optuna import load_study, Study, Trial

from pytorch_tao.args import _ArgSet, arg, arguments

from pytorch_tao.core import ConfigMissingError, ensure_config, load_cfg
from pytorch_tao.repo import DirtyRepoError, Repo
from pytorch_tao.tune import tell


args: _ArgSet = None
study: Study = None
trial: Trial = None
cfg: Any = None
repo: Repo = None

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
    "ConfigMissingError",
    "args",
    "study",
    "trial",
    "DirtyRepoError",
    "arguments",
    "arg",
    "tell",
]
