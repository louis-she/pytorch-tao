import logging
import os
import sys
import tempfile
from pathlib import Path

import optuna

import pytest
import pytorch_tao as tao
from ignite.engine import CallableEventWithFilter, Events

from pytorch_tao import core
from pytorch_tao.exceptions import DirtyRepoError, ConfigMissingError


def test_ensure_config(test_repo: tao.Repo):
    @tao.ensure_config("mount_drive", "dataset_dir")
    def read_colab_drive_file():
        return True

    with pytest.raises(ConfigMissingError) as e:
        read_colab_drive_file()
        assert e.missing_keys == {"mount_drive"}
        assert e.func.__name__ == "read_colab_drive_file"

    os.environ["TAO_ENV"] = "colab"
    tao.load_cfg(test_repo.cfg_path)
    assert read_colab_drive_file()


def test_ensure_arg():
    @tao.ensure_arg("max_epochs")
    def run():
        return True

    with pytest.raises(
        tao.ArgMissingError,
        match="Arg \\{'max_epochs'\\} must be present for calling run",
    ):
        sys.argv = ["mock.py", "--a", "1"]

        class _MockArg_Wrong:
            a: str = 1

        tao.arguments(_MockArg_Wrong)
        run()

    sys.argv = ["mock.py", "--max_epochs", "1"]

    class _MockArg_Right:
        max_epochs: str = 1

    tao.arguments(_MockArg_Right)
    run()


def test_events_register():
    @tao.on(Events.COMPLETED)
    def func():
        pass

    assert func._tao_event == Events.COMPLETED


def test_events_filter_every():
    @tao.on(Events.EPOCH_COMPLETED(every=3))
    def func():
        pass

    assert isinstance(func._tao_event, CallableEventWithFilter)
    assert not func._tao_event.filter(None, 2)
    assert func._tao_event.filter(None, 3)


def test_events_filter_once():
    @tao.on(Events.EPOCH_COMPLETED(once=10))
    def func():
        pass

    assert isinstance(func._tao_event, CallableEventWithFilter)
    assert not func._tao_event.filter(None, 2)
    assert func._tao_event.filter(None, 10)


def test_init_from_env_tao_repo_missing():
    tao.init_from_env()
    assert tao.repo is None
    assert tao.name is None
    assert tao.trial is None
    assert tao.study is None
    assert tao.log_dir is None


def test_init_from_env_tao_repo_exists(test_repo: tao.Repo):
    os.environ["TAO_REPO"] = test_repo.path.as_posix()
    os.environ["TAO_NAME"] = "tao_name"
    tao.init_from_env()
    assert isinstance(tao.repo, tao.Repo)
    assert tao.repo.path == test_repo.path
    assert tao.trial is None
    assert tao.name == "tao_name"
    assert tao.study is None
    assert isinstance(tao.log_dir, Path)


def test_init_from_env_tao_tune(test_repo: tao.Repo):
    os.environ["TAO_REPO"] = test_repo.path.as_posix()
    os.environ["TAO_NAME"] = "tao_name"
    os.environ["TAO_TUNE"] = "1"
    optuna.create_study(
        study_name="tao_name",
        storage=tao.cfg.study_storage,
        direction=tao.cfg.tune_direction,
    )

    tao.init_from_env()
    assert isinstance(tao.repo, tao.Repo)
    assert tao.trial is None
    assert isinstance(tao.study, optuna.Study)
    assert isinstance(tao.log_dir, Path)

    tao_root_logger = logging.getLogger("pytorch_tao")
    assert isinstance(tao_root_logger.handlers[0], logging.StreamHandler)
    assert isinstance(tao_root_logger.handlers[1], logging.FileHandler)
    assert (
        tao_root_logger.handlers[1].baseFilename
        == (tao.cfg.log_dir / "log.txt").absolute().as_posix()
    )
    assert isinstance(tao_root_logger.handlers[2], logging.FileHandler)
    assert (
        tao_root_logger.handlers[2].baseFilename
        == (tao.log_dir / "log.txt").absolute().as_posix()
    )


def test_dispatch_run(test_repo: tao.Repo):
    sys.argv = ["tao", "run", (test_repo.path / "scripts" / "train.py").as_posix()]
    args = core.parse_tao_args()
    assert args.tao_cmd == "run"
    assert (
        args.training_script == (test_repo.path / "scripts" / "train.py").as_posix()
    )

    with pytest.raises(
        DirtyRepoError,
        match="`tao run` requires the repo to be clean, or use `tao run --dirty` to run in dirty mode",
    ):
        core.dispatch(args)


def test_dispatch_tune(test_repo: tao.Repo):
    sys.argv = ["tao", "tune", (test_repo.path / "scripts" / "train.py").as_posix()]
    args = core.parse_tao_args()
    assert args.tao_cmd == "tune"
    assert (
        args.training_script == (test_repo.path / "scripts" / "train.py").as_posix()
    )

    with pytest.raises(
        DirtyRepoError, match="`tao tune` requires the repo to be clean"
    ):
        core.dispatch(args)


def test_dispatch_new():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = Path(tmpdirname)
        project_path = path / "new_tao_project"
        sys.argv = ["tao", "new", project_path.as_posix()]
        args = core.parse_tao_args()
        assert args.tao_cmd == "new"
        assert args.path == project_path.as_posix()
        core.dispatch(args)

        assert project_path.exists()
        assert (project_path / ".tao" / "cfg.py").is_file()
        assert (project_path / ".git").is_dir()
        assert (project_path / ".gitignore").is_file()


def test_dispatch_init():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = Path(tmpdirname)
        sys.argv = ["tao", "init", path.as_posix()]
        args = core.parse_tao_args()
        assert args.tao_cmd == "init"
        assert args.path == path.as_posix()
        core.dispatch(args)

        assert (path / ".tao" / "cfg.py").is_file()
        assert (path / ".git").is_dir()
        assert (path / ".gitignore").is_file()
