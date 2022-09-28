__version__ = "0.1.4"

import os

from optuna import Study, Trial

from pytorch_tao.args import _ArgSet, arg, arguments

from pytorch_tao.core import ConfigMissingError, ensure_config, load_cfg
from pytorch_tao.repo import DirtyRepoError, Repo

if os.getenv("TAO_REPO"):
    repo = Repo(os.getenv("TAO_REPO"))
    cfg = load_cfg(repo.cfg_path)
else:
    repo = None
    cfg = None

args: _ArgSet = None
study: Study = None
trial: Trial = None


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
]
