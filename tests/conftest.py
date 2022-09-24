import os
from pathlib import Path
import shutil
import pytorch_tao as tao
import tempfile
import pytest


@pytest.fixture(autouse=True)
def reset_env():
    os.environ["TAO_ENV"] = ""


@pytest.fixture
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
    mount_drive: true
"""
    repo.cfg_path.write_text(test_cfg)
    tao.load_cfg(repo.cfg_path)
    yield repo
    shutil.rmtree(temp_dir)

