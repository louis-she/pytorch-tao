import argparse
import os
import sys
from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List

import jinja2

from torch.distributed.run import get_args_parser

import pytorch_tao as tao


def load_cfg(cfg_path: Path) -> Dict:
    """Load global config, all the config can be accessed by tao.cfg

    By default, tao will load default cfg, if TAO_ENV is set, tao will load the
    specific part of config. TAO_ENV can be used to seperate different running
    environment, like local machine and colab... All the other part of config will
    inherit from default.
    """
    config_name = os.getenv("TAO_ENV", "default")
    module_name = cfg_path.name.replace(".py", "")
    sys.path.insert(0, cfg_path.parent.as_posix())
    cfg_module = import_module(module_name)
    tao.cfg = getattr(cfg_module, config_name)
    sys.path.pop(0)
    del sys.modules[module_name]


class ConfigMissingError(Exception):
    def __init__(self, keys: List[str], func: Callable):
        self.missing_keys = set(keys)
        self.func = func
        super().__init__(
            f"Config keys {self.missing_keys} must be present for calling {func.__name__}"
        )


def ensure_config(*keys):
    """A decorator to ensure that some config keys must be present for a function"""

    def decorator(func):
        @wraps(func)
        def real(*args, **kwargs):
            missing_keys = [key for key in keys if not hasattr(tao.cfg, key)]
            if len(missing_keys) != 0:
                raise ConfigMissingError(missing_keys, func)
            return func(*args, **kwargs)

        return real

    return decorator


_template_renderer = jinja2.Environment(
    loader=jinja2.PackageLoader("pytorch_tao"), autoescape=jinja2.select_autoescape()
)


def render_tpl(template_name, **kwargs) -> str:
    template = _template_renderer.get_template(template_name + ".jinja")
    return template.render(**kwargs)


def parse_tao_args(args: str = None):
    parser = argparse.ArgumentParser(description="PyTorch Tao")

    subparsers = parser.add_subparsers(dest="tao_cmd")

    run_parser = subparsers.add_parser(
        "run",
        help="Run a training process",
        parents=[get_args_parser()],
        add_help=False,
    )

    run_parser.add_argument(
        "--dirty",
        action="store_true",
        dest="tao_dirty",
        default=False,
        help="If the git repo is not clean, this option should be enabled to start the training, "
        "otherwise Tao will complain about it. Dirty run is for code testing purpose.",
    )

    run_parser.add_argument(
        "--commit",
        type=str,
        dest="tao_commit",
        default=None,
        help="Commit and run the code, it is equal to `git add -A; git commit -m xxx; tao run xxx`",
    )

    new_parser = subparsers.add_parser(
        "new",
        help="Create a tao project",
    )

    new_parser.add_argument("path", type=str, help="Path of this new project")

    tune_parser = subparsers.add_parser(
        "tune",
        parents=[get_args_parser()],
        help="Tune parameters of a training script",
        add_help=False,
    )

    tune_parser.add_argument(
        "--name", type=str, dest="tao_tune_name", help="Study name"
    )
    tune_parser.add_argument(
        "--max_trials", type=int, dest="tao_tune_max_trials", help="Number of trails"
    )
    tune_parser.add_argument(
        "--duplicated",
        action="store_true",
        dest="tao_tune_duplicated",
        help="Is this study name already used and want to resume?",
    )

    tao.args = parser.parse_args(args)


def run():
    tao.repo = tao.Repo.find_by_file(tao.args.training_script)
    tao.load_cfg(tao.repo.cfg_path)
    tao.repo.run()


def new():
    tao.Repo.create(tao.args.path)


def tune():
    tao.repo = tao.Repo.find_by_file(tao.args.training_script)
    tao.load_cfg(tao.repo.cfg_path)
    tao.repo.tune()


_cmd = {
    "run": run,
    "new": new,
    "tune": tune,
}


def dispatch():
    if tao.args is None:
        raise RuntimeError("Should parse args before dispatch")
    _cmd[tao.args.tao_cmd]()
