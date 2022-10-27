import os
from unittest.mock import DEFAULT, patch

import pytest
import pytorch_tao as tao
from pytorch_tao import exceptions
import torch
import wandb
from pytorch_tao.trackers.wandb_tracker import WandbTracker


def test_wandb_tracker_init(test_repo):
    with pytest.raises(
        exceptions.ConfigMissingError,
        match="Config keys \\{'wandb_project'\\} must be present for calling __init__",
    ):
        WandbTracker("random_name")

    tao.cfg.wandb_project = "some project"

    with pytest.raises(wandb.errors.UsageError, match="api_key not configured"):
        WandbTracker("random_name")

    tao.cfg.wandb_api_key = "123456"
    with patch.object(wandb, "init") as wandb_init:
        WandbTracker("random_name")
        assert os.environ["WANDB_API_KEY"] == "123456"
        wandb_init.assert_called_once_with(
            dir=tao.cfg.log_dir,
            project=tao.cfg.wandb_project,
            name="random_name",
            group=None,
        )


def test_wandb_tracker_add_image(test_repo):
    tao.cfg.wandb_project = "some project"
    tao.cfg.wandb_api_key = "123456"

    with patch.multiple(wandb, init=DEFAULT, log=DEFAULT) as values:
        tracker = WandbTracker("random_name")
        image = torch.rand((3, 512, 512))
        tracker.add_image("vis", [image])
        values["log"].assert_called_once_with({"vis": [wandb.Image(image)]})

    with patch.multiple(wandb, init=DEFAULT, log=DEFAULT) as values:
        tracker = WandbTracker("random_name")
        image = torch.rand((3, 512, 512))
        tracker.add_image("vis", image)
        values["log"].assert_called_once_with({"vis": [wandb.Image(image)]})


def test_wandb_tracker_add_points(test_repo):
    tao.cfg.wandb_project = "some project"
    tao.cfg.wandb_api_key = "123456"

    with patch.multiple(wandb, init=DEFAULT, log=DEFAULT) as values:
        tracker = WandbTracker("random_name")
        point = {
            "x": 1,
            "y": 2,
        }
        tracker.add_points(point)
        values["log"].assert_called_once_with(point)
