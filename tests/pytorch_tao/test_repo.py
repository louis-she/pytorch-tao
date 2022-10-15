import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import optuna

import pytest
import pytorch_tao as tao
from pytorch_tao import core


def test_create_repo():
    with pytest.raises(FileNotFoundError):
        tao.Repo.create("/a/random/not/existsed/dir")

    with pytest.raises(FileExistsError):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tao.Repo.create(tmpdir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo = tao.Repo.create(tmpdir / "random_project")
        assert repo.exists()
        assert repo.git.head.ref.commit.message == "initial commit"
        assert repo.tao_path.is_dir()
        assert repo.cfg_path.is_file()
        assert (repo.path / ".gitignore").is_file()
    assert not repo.exists()


def test_find_repo_by_file(test_repo: tao.Repo):
    subdir = test_repo.path / "sub1" / "sub2" / "sub3"
    subdir.mkdir(parents=True)
    subfile = subdir / "some_file.txt"
    subfile.touch()

    assert tao.Repo.find_by_file(subdir).name == test_repo.name
    assert tao.Repo.find_by_file(subfile).name == test_repo.name

    with pytest.raises(FileNotFoundError):
        with tempfile.NamedTemporaryFile() as file:
            tao.Repo.find_by_file(file.name)


def test_load_config(test_repo: tao.Repo):
    assert tao.cfg.dataset_dir == "/dataset/dir/in/default/cfg"
    assert tao.cfg.kaggle_username == "snaker"
    assert tao.cfg.kaggle_key == "xxxxxx"
    assert not hasattr(tao.cfg, "mount_drive")

    os.environ["TAO_ENV"] = "colab"
    tao.load_cfg(test_repo.cfg_path)
    assert tao.cfg.dataset_dir == "/dataset/dir/in/colab/cfg"
    assert tao.cfg.kaggle_username == "snaker"
    assert tao.cfg.kaggle_key == "xxxxxx"
    assert tao.cfg.mount_drive


def test_sync_code_to_kaggle(test_repo: tao.Repo):
    # set the env or import kaggle will raise error
    os.environ["KAGGLE_USERNAME"] = "xxxxxx"
    os.environ["KAGGLE_KEY"] = "xxxxxx"
    import kaggle

    kaggle.api.dataset_create_version = MagicMock(return_value=True)
    test_repo.sync_code_to_kaggle()
    kaggle.api.dataset_create_version.assert_called_once()


def test_run_dirty(test_repo: tao.Repo):
    with pytest.raises(tao.DirtyRepoError):
        command = f"run {(test_repo.path / 'scripts' / 'train.py').as_posix()} --test --epochs 10".split(
            " "
        )
        core.parse_tao_args(command)
        test_repo.run()


def option_value(argv, option):
    return argv[argv.index(option) + 1]


def test_run_with_dirty_option(test_repo: tao.Repo):
    command = f"run --dirty {(test_repo.path / 'scripts' / 'train.py').as_posix()} --test --epochs 10".split(
        " "
    )
    core.parse_tao_args(command)
    test_repo.run()
    with (test_repo.path / "result.json").open("r") as f:
        result = json.load(f)

    assert result["local_rank"] == "0"
    assert result["cwd"] == test_repo.path.as_posix()
    assert result["some_lib_path"] == (test_repo.path / "some_lib.py").as_posix()
    assert (
        result["some_package_path"]
        == (test_repo.path / "some_package" / "__init__.py").as_posix()
    )

    argv = result["argv"]
    assert "--test" in argv
    assert "--epochs" in argv
    assert "10" == option_value(argv, "--epochs")
    assert "--dirty" not in argv
    assert "--commit" not in argv


def test_run_clean_repo(test_repo: tao.Repo):
    test_repo.git.git.add(all=True)
    test_repo.git.index.commit("clean dirty")

    command = (
        f"run {(test_repo.path / 'scripts' / 'train.py').as_posix()} --test --epochs 10"
    )
    command = command.split(" ")
    core.parse_tao_args(command)
    test_repo.run()

    hash = test_repo.git.head.ref.commit.hexsha[:8]
    run_dir = test_repo.path / "runs" / hash
    assert (run_dir / "result.json").is_file()
    assert (run_dir / ".tao").is_dir()
    assert (run_dir / "scripts" / "train.py").is_file()
    assert (run_dir / "some_package" / "__init__.py").is_file()

    with (run_dir / "result.json").open("r") as f:
        result = json.load(f)

    argv = result["argv"]
    assert "--dirty" not in argv
    assert result["some_lib_path"] == (run_dir / "some_lib.py").as_posix()
    assert (
        result["some_package_path"]
        == (run_dir / "some_package" / "__init__.py").as_posix()
    )


def test_run_commit(test_repo: tao.Repo):
    command = f"run --commit some_comments {(test_repo.path / 'scripts' / 'train.py').as_posix()} --test --epochs 10"
    command = command.split(" ")
    core.parse_tao_args(command)
    test_repo.run()
    assert not test_repo.git.is_dirty()
    assert test_repo.git.head.ref.commit.message == "some_comments"


def test_run_with_arguments(test_repo_with_arguments: tao.Repo):
    command = (
        f"run {(test_repo_with_arguments.path / 'main.py').as_posix()} "
        f"--trial_name test --max_epochs 10 --train_folds 1 2 3"
    )
    command = command.split(" ")
    core.parse_tao_args(command)
    test_repo_with_arguments.run()

    hash = test_repo_with_arguments.git.head.ref.commit.hexsha[:8]
    run_dir = test_repo_with_arguments.path / ".tao" / "runs" / hash
    with (run_dir / "args.json").open("r") as f:
        result = json.load(f)
    assert result["trial_name"] == "test"
    assert result["optimizer"] is None
    assert result["train_folds"] == [1, 2, 3]
    assert result["val_folds"] == [0]
    assert result["model"] == "resnet34"
    assert result["max_epochs"] == 10
    assert result["batch_size"] == 32


def test_tune(test_repo_for_tune: tao.Repo):
    command = (
        "tune --name test_tune --max_trials 10 "
        f"{(test_repo_for_tune.path / 'main.py').as_posix()}"
    )
    command = command.split(" ")
    core.parse_tao_args(command)
    test_repo_for_tune.tune()
    study = optuna.load_study(
        study_name="tune_test_tune",
        storage=f"sqlite:////{(test_repo_for_tune.tao_path / 'study.db').as_posix()}",
    )
    assert len(study.trials) == 10
