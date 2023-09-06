"""
PyTorch Tao
"""


__version__ = "0.1.10"

import logging
from pathlib import Path
from typing import Any

from optuna import Study, Trial

from pytorch_tao import helper

from pytorch_tao.args import _ArgSet, arg, arguments

from pytorch_tao.core import (
    ArgMissingError,
    ensure_arg,
    ensure_config,
    init_from_env,
    load_cfg,
    on,
)
from pytorch_tao.repo import Repo
from pytorch_tao.trackers import set_tracker, Tracker
from pytorch_tao.trainer import Trainer
from pytorch_tao.tune import tell

logging.captureWarnings(True)

args: _ArgSet = None
"""
``tao.args`` contains the value of arguments defined with ``tao.arguments``, access them like ``tao.args.xxx``.

..  code-block:: python

    @tao.arguments
    class _:
        learning_rate: int = tao.arg(default=3e-4)

    assert tao.args.learning_rate == 3e-4


More example can be found in tests.

"""

cfg: Any = None
"""
Object that hold all the config reading from config file(by default :code:`.tao/cfg.py`).
We can access the config with :code:`tao.cfg.xxx`.
"""

tracker = Tracker()
"""
:class:`pytorch_tao.trackers.base.Tracker` object that is set by :func:`~pytorch_tao.trackers.set_tracker`.
"""

log_dir: Path = None
"""
:class:`pathlib.Path` object that represent the log directory of the process.
"""

name: str = None
"""
str that represent name of this process, useful for seperating different runs.
"""

tune: bool = False
"""
boolean that indicates if current run is in tune mode.
"""

repo: Repo = None
"""
:py:class:`pytorch_tao.repo.Repo` object that represent current code executing repo.

.. note::
    Normally this variable is only used by **Tao** internally or plugins.

"""

study: Study = None
"""
:py:class:`optuna:optuna.study.Study` object which can be used in tune mode.
"""

trial: Trial = None
"""
:py:class:`optuna:optuna.trial.Trial` object which can be used in tune mode.
"""


init_from_env()


def get_logger(name):
    return logging.getLogger(f"pytorch_tao.{name}")


__all__ = [
    "Repo",
    "load_cfg",
    "cfg",
    "ensure_config",
    "ensure_arg",
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
    "get_logger",
    "helper",
]
