import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytorch_tao as tao


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
    import kaggle

    kaggle.api.dataset_create_version = MagicMock(return_value=True)
    test_repo.sync_code_to_kaggle()
    kaggle.api.dataset_create_version.assert_called_once()
