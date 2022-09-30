import pytest
from pytorch_tao.plugins import base


@pytest.fixture(scope="module")
def empty_base_plugin():
    base_plugin_klass = type("a_base_plugin", (base.BasePlugin,), {})
    return base_plugin_klass()


@pytest.fixture(scope="module")
def empty_train_plugin():
    train_plugin_klass = type("a_train_plugin", (base.TrainPlugin,), {})
    return train_plugin_klass()


@pytest.fixture(scope="module")
def empty_val_plugin():
    val_plugin_klass = type("a_val_plugin", (base.ValPlugin,), {})
    return val_plugin_klass()


def test_base_plugin_init(empty_base_plugin: base.TrainPlugin):
    assert empty_base_plugin.attach_to is None


def test_train_plugin_init(empty_train_plugin: base.TrainPlugin):
    assert empty_train_plugin.attach_to == "train_engine"


def test_val_plugin_init(empty_val_plugin: base.ValPlugin):
    assert empty_val_plugin.attach_to == "val_engine"
