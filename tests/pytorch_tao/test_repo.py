import io
import json
import os
import subprocess
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

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
    assert tao.cfg["dataset_dir"] == "/dataset/dir/in/default/cfg"
    assert tao.cfg["kaggle_username"] == "snaker"
    assert tao.cfg["kaggle_key"] == "xxxxxx"
    assert "mount_drive" not in tao.cfg

    os.environ["TAO_ENV"] = "colab"
    tao.load_cfg(test_repo.cfg_path)
    assert tao.cfg["dataset_dir"] == "/dataset/dir/in/colab/cfg"
    assert tao.cfg["kaggle_username"] == "snaker"
    assert tao.cfg["kaggle_key"] == "xxxxxx"
    assert tao.cfg["mount_drive"]


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
        core.parse_args(command)
        test_repo.run()


def test_run_with_dirty_option(test_repo: tao.Repo):
    command = f"run --dirty {(test_repo.path / 'scripts' / 'train.py').as_posix()} --test --epochs 10".split(
        " "
    )
    core.parse_args(command)
    test_repo.run()
    with (test_repo.path / "result.json").open("r") as f:
        result = json.load(f)
    assert result["local_rank"] == "0"
    assert "--test" in result["argv"]
    assert "--dirty" in result["argv"]
    assert "--commit" in result["argv"]
    assert result["cwd"] == test_repo.path.as_posix()
    assert result["some_lib_path"] == (test_repo.path / "some_lib.py").as_posix()
    assert (
        result["some_package_path"]
        == (test_repo.path / "some_package" / "__init__.py").as_posix()
    )
