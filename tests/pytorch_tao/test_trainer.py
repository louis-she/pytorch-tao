import pytorch_tao as tao
from ignite.engine import Events
from pytorch_tao.plugins.base import TrainPlugin, ValPlugin


class Counter:
    def __init__(self):
        super().__init__()
        self.count = 0

    @tao.on(Events.EPOCH_STARTED)
    def _add_count(self):
        self.count += 1


class TestTrainPlugin(Counter, TrainPlugin):
    pass


class TestValPlugin(Counter, ValPlugin):
    pass


def test_use_train_plugin(trainer: tao.Trainer):
    train_plugin = TestTrainPlugin()
    trainer.use(train_plugin)
    trainer.fit(max_epochs=5)
    assert train_plugin.count == 5


def test_use_val_plugin(trainer: tao.Trainer):
    val_plugin = TestValPlugin()
    trainer.use(val_plugin)
    trainer.fit(max_epochs=5)
    assert val_plugin.count == 5
