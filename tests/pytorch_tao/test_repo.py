import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest
import pytorch_tao as tao


@contextlib.contextmanager
def test_repo():
    temp_dir = Path(tempfile.mkdtemp())
    repo_dir = temp_dir / "test_repo"
    repo = tao.Repo.create(repo_dir)
    test_cfg = """
default:
    dataset_dir: /dataset/dir/in/default/cfg
    kaggle_username: snaker
    kaggle_key: xxxxxx
colab:
    dataset_dir: /dataset/dir/in/colab/cfg
"""
    repo.cfg_path.write_text(test_cfg)
    tao.load_cfg(repo.cfg_path)
    yield repo
    shutil.rmtree(temp_dir)


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
    assert not repo.exists()


def test_load_config():
    with test_repo():
        assert tao.cfg["dataset_dir"] == "/dataset/dir/in/default/cfg"
        assert tao.cfg["kaggle_username"] == "snaker"
        assert tao.cfg["kaggle_key"] == "xxxxxx"

    os.environ["TAO_ENV"] = "colab"
    with test_repo():
        assert tao.cfg["dataset_dir"] == "/dataset/dir/in/colab/cfg"
        assert tao.cfg["kaggle_username"] == "snaker"
        assert tao.cfg["kaggle_key"] == "xxxxxx"
