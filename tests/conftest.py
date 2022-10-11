import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import jinja2
import numpy as np
import pytest
import pytorch_tao as tao
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ignite.metrics import Metric as IMetric
from torch.optim import SGD


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class SumMetric(IMetric):
    def __init__(self, *args, **kwargs):
        self._start = -1
        super().__init__(*args, **kwargs)

    def reset(self):
        self._start += 1
        self._sum = self._start

    def update(self, output):
        self._sum += output

    def compute(self):
        return self._sum


@pytest.fixture
def simplenet():
    return SimpleNet()


@pytest.fixture(scope="function")
def sum_metric():
    return SumMetric()


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
def trainer():
    return tao.Trainer(
        train_func=lambda e, b: 1,
        train_loader=range(100),
        val_func=lambda e, b: 1,
        val_loader=range(100),
    )


@pytest.fixture(scope="function")
def fake_mnist_trainer():
    dataset = torchvision.datasets.FakeData(
        size=100, image_size=(1, 28, 28), transform=torchvision.transforms.ToTensor()
    )
    train_set = torch.utils.data.Subset(dataset, range(0, 80))
    val_set = torch.utils.data.Subset(dataset, range(80, 100))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4)

    model = SimpleNet()
    trainer = tao.Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
    )

    return trainer


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


@pytest.fixture(scope="function")
def test_repo_for_tune(render_tpl):
    temp_dir = Path(tempfile.mkdtemp())
    repo_dir = temp_dir / "test_repo_for_tune"
    repo = tao.Repo.create(repo_dir)
    (repo.path / "main.py").write_text(render_tpl("repo_for_tune_main.py"))
    (repo.path / ".tao" / "cfg.py").write_text(
        render_tpl(
            "repo_for_tune_cfg.py",
            sqlite_storage_path=(repo.tao_path / "study.db").as_posix(),
            run_dir=(repo.tao_path / "runs").as_posix(),
        )
    )
    tao.load_cfg(repo.cfg_path)
    repo.commit_all("add all")
    yield repo
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def tracker():

    _previous_tracker = tao.tracker

    class _Tracker(tao.Tracker):
        def __init__(self):
            super().__init__()
            self.points = []
            self.images = []

        def add_points(self, points: Dict):
            super().add_points(points)
            self.points.append(points)

        def add_image(self, image: np.ndarray):
            super().add_image(image)
            self.images.append(image)

    tao.tracker = _Tracker()
    yield tao.tracker

    tao.tracker = _previous_tracker
