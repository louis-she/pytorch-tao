import os
import shutil
import tempfile
from pathlib import Path

import jinja2

import pytest
import pytorch_tao as tao


@pytest.fixture(autouse=True)
def reset_env():
    try:
        os.environ.pop("TAO_ENV")
    except KeyError:
        pass


@pytest.fixture(scope="session")
def render_tpl():
    template_renderer = jinja2.Environment(
        loader=jinja2.PackageLoader("tests"), autoescape=jinja2.select_autoescape()
    )

    def render_tpl(template_name, **kwargs) -> str:
        template = template_renderer.get_template(template_name + ".jinja")
        return template.render(**kwargs)

    return render_tpl


@pytest.fixture(scope="function")
def test_repo():
    temp_dir = Path(tempfile.mkdtemp())
    repo_dir = temp_dir / "test_repo"
    repo = tao.Repo.create(repo_dir)
    test_cfg = """
class default:
    run_dir = "./runs/"
    dataset_dir = "/dataset/dir/in/default/cfg"
    kaggle_username = "snaker"
    kaggle_key = "xxxxxx"
    kaggle_dataset_id = "the_dataset_id"

class colab(default):
    dataset_dir = "/dataset/dir/in/colab/cfg"
    mount_drive = True
"""
    repo.cfg_path.write_text(test_cfg)
    tao.load_cfg(repo.cfg_path)

    (repo.path / "some_lib.py").touch()

    (repo.path / "some_package").mkdir()
    (repo.path / "some_package" / "__init__.py").touch()

    (repo.path / "scripts").mkdir()
    (repo.path / "scripts" / "train.py").write_text(
        """
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


@pytest.fixture(scope="function")
def test_repo_with_arguments(render_tpl):
    temp_dir = Path(tempfile.mkdtemp())
    repo_dir = temp_dir / "test_repo_with_arguments"
    repo = tao.Repo.create(repo_dir)
    (repo.path / "main.py").write_text(render_tpl("repo_with_arguments_main.py"))
    repo.commit_all("add main.py")
    yield repo
    shutil.rmtree(temp_dir)
