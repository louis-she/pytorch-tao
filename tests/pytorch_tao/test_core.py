import os
from pathlib import Path

import pytest
import pytorch_tao as tao
from ignite.engine import CallableEventWithFilter, Events
import optuna


def test_ensure_config(test_repo: tao.Repo):
    @tao.ensure_config("mount_drive", "dataset_dir")
    def read_colab_drive_file():
        return True

    with pytest.raises(tao.ConfigMissingError) as e:
        read_colab_drive_file()
        assert e.missing_keys == {"mount_drive"}
        assert e.func.__name__ == "read_colab_drive_file"

    os.environ["TAO_ENV"] = "colab"
    tao.load_cfg(test_repo.cfg_path)
    assert read_colab_drive_file()


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
