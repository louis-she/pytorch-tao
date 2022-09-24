import os
import shutil
import tempfile
from pathlib import Path

import pytest
import pytorch_tao as tao


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
    kaggle_dataset_id: the_dataset_id
colab:
    dataset_dir: /dataset/dir/in/colab/cfg
    mount_drive: true
"""
    repo.cfg_path.write_text(test_cfg)
    tao.load_cfg(repo.cfg_path)
    yield repo
    shutil.rmtree(temp_dir)
