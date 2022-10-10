import pytest
import pytorch_tao as tao
import torch
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


def test_trainer_train_decorator(fake_mnist_trainer: tao.Trainer):

    @fake_mnist_trainer.train()
    def _(images, labels):
        assert images.shape == torch.Size((4, 1, 28, 28))
        assert labels.shape == torch.Size((4,))
        assert images.device.type == "cpu"
        assert labels.device.type == "cpu"

    fake_mnist_trainer.fit(max_epochs=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_trainer_train_decorator_cuda_device(fake_mnist_trainer: tao.Trainer):

    fake_mnist_trainer.device = torch.device("cuda")

    @fake_mnist_trainer.train()
    def _(images, labels):
        assert images.device.type == "cuda"
        assert labels.device.type == "cuda"

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_custom_fields(fake_mnist_trainer: tao.Trainer):

    @fake_mnist_trainer.train(fields=[0])
    def _(images, *args):
        assert images.shape == torch.Size((4, 1, 28, 28))
        assert len(args) == 0

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_no_grad(fake_mnist_trainer: tao.Trainer):

    @fake_mnist_trainer.train(grad=False)
    def _(images, labels):
        logits = fake_mnist_trainer.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        fake_mnist_trainer.train_engine.should_terminate = True

    with pytest.raises(RuntimeError, match="element 0 of tensors does not require grad and does not have a grad_fn"):
        fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_grad(fake_mnist_trainer: tao.Trainer):

    @fake_mnist_trainer.train(grad=True)
    def _(images, labels):
        logits = fake_mnist_trainer.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        fake_mnist_trainer.train_engine.should_terminate = True
    fake_mnist_trainer.fit(max_epochs=1)
