import pytest
import pytorch_tao as tao
from ignite.engine import Events
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
    assert empty_train_plugin.attach_to == "train"


def test_val_plugin_init(empty_val_plugin: base.ValPlugin):
    assert empty_val_plugin.attach_to == "val"


def test_attach_events(trainer: tao.Trainer):
    class _TestPlugin(base.TrainPlugin):
        def __init__(self):
            super().__init__()
            self.sum = 0
            self.sum_with_stride = 0
            self.sum_stride = 2

        @tao.on(Events.ITERATION_COMPLETED)
        def _plus_one(self):
            self.sum += 1

        @tao.on(lambda self: Events.ITERATION_COMPLETED(every=self.sum_stride))
        def _plus_one_stride(self):
            self.sum_with_stride += 1

    plugin = _TestPlugin()
    trainer.use(plugin)
    trainer.fit(max_epochs=1)

    assert plugin.sum == 100
    assert plugin.sum_with_stride == 50


def test_attach_wrong_events(trainer: tao.Trainer):
    class _TestPlugin(base.TrainPlugin):
        @tao.on(1)
        def _(self):
            pass

    class _TestPlugin_2(base.TrainPlugin):
        @tao.on(lambda self: 1)
        def _(self):
            pass

    class _TestPlugin_3(base.TrainPlugin):
        @tao.on(lambda self: Events.ITERATION_COMPLETED(every=2))
        def _(self):
            pass

    with pytest.warns(
        Warning, match="event handler lambda should return valid event type"
    ):
        trainer.use(_TestPlugin())

    with pytest.warns(
        Warning, match="event handler lambda should return valid event type"
    ):
        trainer.use(_TestPlugin_2())

    trainer.use(_TestPlugin_3())
