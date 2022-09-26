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
    run_dir: ./runs/
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

    (repo.path / "some_lib.py").touch()

    (repo.path / "some_package").mkdir()
    (repo.path / "some_package" / "__init__.py").touch()

    (repo.path / "scripts").mkdir()
    (repo.path / "scripts" / "train.py").write_text("""
import os
import sys
import json
import some_lib
import some_package
import pytorch_tao as tao

with (tao.repo.path / "result.json").open("w") as f:
    json.dump({
        "local_rank": os.environ["LOCAL_RANK"],
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "some_lib_path": some_lib.__file__,
        "some_package_path": some_package.__file__,
    }, f)
"""
    )

    yield repo
    shutil.rmtree(temp_dir)
