import os

import pytest
import pytorch_tao as tao
from ignite.engine import CallableEventWithFilter, Events


def test_ensure_config(test_repo: tao.Repo):
    @tao.ensure_config("mount_drive", "dataset_dir")
    def read_colab_drive_file():
        return True

    with pytest.raises(tao.ConfigMissingError) as e:
        read_colab_drive_file()
        assert e.missing_keys == {"mount_drive"}
        assert e.func.__name__ == "read_colab_drive_file"

    os.environ["TAO_ENV"] = "colab"
    tao.load_cfg(test_repo.cfg_path)
    assert read_colab_drive_file()


def test_events_register():
    @tao.on(Events.COMPLETED)
    def func():
        pass

    assert func._tao_event == Events.COMPLETED


def test_events_filter_every():
    @tao.on(Events.EPOCH_COMPLETED(every=3))
    def func():
        pass

    assert isinstance(func._tao_event, CallableEventWithFilter)
    assert not func._tao_event.filter(None, 2)
    assert func._tao_event.filter(None, 3)


def test_events_filter_once():
    @tao.on(Events.EPOCH_COMPLETED(once=10))
    def func():
        pass

    assert isinstance(func._tao_event, CallableEventWithFilter)
    assert not func._tao_event.filter(None, 2)
    assert func._tao_event.filter(None, 10)
