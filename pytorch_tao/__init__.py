__version__ = "0.1.4"

import os
from argparse import Namespace

from pytorch_tao.core import ConfigMissingError, ensure_config, load_cfg
from pytorch_tao.repo import DirtyRepoError, Repo
from pytorch_tao.args import arguments


args: Namespace = None

if os.getenv("TAO_REPO"):
    repo = Repo(os.getenv("TAO_REPO"))
    cfg = load_cfg(repo.cfg_path)
else:
    repo = None
    cfg = None

args = None


__all__ = [
    "Repo",
    "load_cfg",
    "cfg",
    "ensure_config",
    "ConfigMissingError",
    "args",
    "DirtyRepoError",
    "arguments"
]
