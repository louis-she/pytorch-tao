import argparse
import logging
import os
import sys
from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List

import ignite.distributed as idist

import jinja2
from optuna import load_study

from torch.distributed.run import get_args_parser

import pytorch_tao as tao
from pytorch_tao import exceptions


_log_format = "[%(levelname).1s %(asctime)s] %(message)s"


@idist.one_rank_only()
class StreamLogFormatter(logging.Formatter):
    grey = "\033[0;37m"
    green = "\033[0;32m"
    yellow = "\033[1;33m"
    red = "\033[0;31m"
    reset = "\033[0m"

    FORMATS = {
        logging.DEBUG: grey + "%s" + reset,
        logging.INFO: green + "%s" + reset,
        logging.WARNING: yellow + "%s" + reset,
        logging.ERROR: red + "%s" + reset,
        logging.CRITICAL: red + "%s" + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        return log_fmt % super().format(record)


def init_logger():
    for logger in [logging.getLogger("pytorch_tao"), logging.getLogger("py.warnings")]:
        logger.setLevel(logging.INFO)
        one_rank_only = idist.one_rank_only()

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.emit = one_rank_only(stream_handler.emit)
        stream_handler.setFormatter(StreamLogFormatter(fmt=_log_format))

        file_formatter = logging.Formatter(fmt=_log_format)
        main_file_handler = logging.FileHandler(tao.cfg.log_dir / "log.txt", mode="a")
        main_file_handler.emit = one_rank_only(main_file_handler.emit)
        main_file_handler.setFormatter(file_formatter)

        run_file_handler = logging.FileHandler(tao.log_dir / "log.txt", mode="w")
        run_file_handler.emit = one_rank_only(run_file_handler.emit)
        run_file_handler.setFormatter(file_formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(main_file_handler)
        logger.addHandler(run_file_handler)

    logging.captureWarnings(True)


def init_from_env():
    if not os.getenv("TAO_REPO", None):
        return

    from pytorch_tao.repo import Repo

    # tao.repo, tao.cfg
    tao.repo = Repo(os.getenv("TAO_REPO"))

    # tao.name
    tao.name = os.getenv("TAO_NAME")

    # tao.tune, tao.study
    tao.tune = True if os.getenv("TAO_TUNE") else False
    if tao.tune:
        tao.study = load_study(study_name=tao.name, storage=tao.cfg.study_storage)
        # tao.trial will be inited in argument parse

    # tao.log_dir
    tao.log_dir = Path(tao.cfg.log_dir) / tao.name
    tao.log_dir.mkdir(exist_ok=True, parents=True)

    init_logger()


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


class ArgMissingError(Exception):
    def __init__(self, keys: List[str], func: Callable):
        self.missing_keys = set(keys)
        self.func = func
        super().__init__(
            f"Arg {self.missing_keys} must be present for calling {func.__name__}"
        )


def ensure_arg(*keys):
    """A decorator to ensure that some config keys must be present for a function"""

    def decorator(func):
        @wraps(func)
        def real(*args, **kwargs):
            missing_keys = [
                key
                for key in keys
                if not hasattr(tao.args, key) or getattr(tao.args, key) is None
            ]
            if len(missing_keys) != 0:
                raise ArgMissingError(missing_keys, func)
            return func(*args, **kwargs)

        return real

    return decorator


def ensure_config(*keys):
    """A decorator to ensure that some config keys must be present for a function"""

    def decorator(func):
        @wraps(func)
        def real(*args, **kwargs):
            missing_keys = [
                key
                for key in keys
                if not hasattr(tao.cfg, key) or getattr(tao.cfg, key) is None
            ]
            if len(missing_keys) != 0:
                raise exceptions.ConfigMissingError(missing_keys, func)
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

    kaggle_parser = subparsers.add_parser(
        "sync_kaggle", help="Sync repo code to kaggle dataset"
    )

    kaggle_parser.add_argument(
        "path", type=str, default=".", nargs="?", help="Path of the existing project"
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a training process",
        parents=[get_args_parser()],
        add_help=False,
    )

    run_parser.add_argument(
        "--name",
        type=str,
        dest="tao_name",
        default=None,
        help="Name of the this run",
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
        "--checkout",
        type=str,
        dest="tao_checkout",
        default=None,
        help="Run a specific commit(or branch or tag)",
    )

    new_parser = subparsers.add_parser(
        "new",
        help="Create a tao project",
    )

    new_parser.add_argument(
        "--template",
        type=str,
        dest="tao_template",
        help="Template to init the new project",
        default="mini",
    )
    new_parser.add_argument("path", type=str, help="Path of this new project")

    init_parser = subparsers.add_parser(
        "init",
        help="Init a tao project",
    )

    init_parser.add_argument(
        "--template",
        type=str,
        dest="tao_template",
        help="Template to init the new project",
        default="mini",
    )

    init_parser.add_argument(
        "path", type=str, default=".", nargs="?", help="Path of the existing project"
    )

    tune_parser = subparsers.add_parser(
        "tune",
        parents=[get_args_parser()],
        help="Tune parameters of a training script",
        add_help=False,
    )

    tune_parser.add_argument(
        "--name",
        type=str,
        dest="tao_tune_name",
        help="Name of the this tune, also will be used as study name",
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
    tune_parser.add_argument(
        "--checkout",
        type=str,
        dest="tao_tune_checkout",
        default=None,
        help="Tune a specific commit(or branch or tag)",
    )

    return parser.parse_args(args)


def dispatch(args: argparse.Namespace):
    def _run():
        tao.repo = tao.Repo.find_by_file(args.training_script)
        tao.repo.run(args.tao_name, args.tao_dirty, args.tao_checkout)

    def _new():
        tao.Repo.create(path=args.path, template=args.tao_template)

    def _init():
        repo = tao.Repo(args.path)
        repo.init(args.tao_template)

    def _tune():
        tao.repo = tao.Repo.find_by_file(args.training_script)
        tao.load_cfg(tao.repo.cfg_path)
        tao.repo.tune(
            args.tao_tune_name, args.tao_tune_max_trials, args.tao_tune_duplicated
        )

    def _sync_kaggle():
        repo = tao.Repo(args.path)
        repo.sync_code_to_kaggle()

    _cmd = {
        "run": _run,
        "new": _new,
        "init": _init,
        "tune": _tune,
        "sync_kaggle": _sync_kaggle,
    }

    _cmd[args.tao_cmd]()


def on(event: Callable):
    def decorator(func):
        func._tao_event = event
        return func

    return decorator
