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
        assert next(fake_mnist_trainer.model.parameters()).device.type == "cpu"
        assert fake_mnist_trainer.model.training
        return torch.tensor(1.0, requires_grad=True)

    fake_mnist_trainer.fit(max_epochs=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_trainer_train_decorator_cuda_device(fake_mnist_trainer: tao.Trainer):

    fake_mnist_trainer.to("cuda")

    @fake_mnist_trainer.train()
    def _(images, labels):
        assert images.device.type == "cuda"
        assert labels.device.type == "cuda"
        assert next(fake_mnist_trainer.model.parameters()).device.type == "cuda"
        return torch.tensor(1.0, requires_grad=True)

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_custom_fields(fake_mnist_trainer: tao.Trainer):
    @fake_mnist_trainer.train(fields=[0])
    def _(images, *args):
        assert images.shape == torch.Size((4, 1, 28, 28))
        assert len(args) == 0
        return torch.tensor(1.0, requires_grad=True)

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_no_grad(fake_mnist_trainer: tao.Trainer):
    @fake_mnist_trainer.train(grad=False)
    def _(images, labels):
        logits = fake_mnist_trainer.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        fake_mnist_trainer.train_engine.should_terminate = True
        return loss

    with pytest.raises(
        RuntimeError,
        match="element 0 of tensors does not require grad and does not have a grad_fn",
    ):
        fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_grad(
    fake_mnist_trainer: tao.Trainer, tracker: tao.Tracker
):
    @fake_mnist_trainer.train(grad=True)
    def _(images, labels):
        logits = fake_mnist_trainer.model(images)
        return torch.nn.functional.cross_entropy(logits, labels)

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_amp(
    fake_mnist_trainer: tao.Trainer, tracker: tao.Tracker
):
    @fake_mnist_trainer.train(amp=False)
    def _(images, labels):
        output = torch.tensor([1.0]) @ torch.tensor([2.0])
        assert output.dtype == torch.float32
        return torch.tensor(1.0, requires_grad=True)

    fake_mnist_trainer.fit(max_epochs=1)

    @fake_mnist_trainer.train(amp=True)
    def _(images, labels):
        output = torch.tensor([1.0]) @ torch.tensor([2.0])
        assert output.dtype == torch.bfloat16
        return torch.tensor(1.0, requires_grad=True)

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_output_scalar(
    fake_mnist_trainer: tao.Trainer, tracker: tao.Tracker
):
    @fake_mnist_trainer.train(amp=False)
    def _(images, labels):
        return torch.tensor(1.0, requires_grad=False)

    with pytest.raises(
        RuntimeError,
        match="element 0 of tensors does not require grad and does not have a grad_fn",
    ):
        fake_mnist_trainer.fit(max_epochs=1)

    @fake_mnist_trainer.train(amp=True)
    def _(images, labels):
        return torch.tensor(1.0, requires_grad=False)

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_output_list(
    fake_mnist_trainer: tao.Trainer, tracker: tao.Tracker
):
    @fake_mnist_trainer.train()
    def _(images, labels):
        return [
            torch.tensor(1.0, requires_grad=False),
            torch.tensor(2.0, requires_grad=True),
            1,
            "some_other_thing",
        ]

    with pytest.raises(
        RuntimeError,
        match="element 0 of tensors does not require grad and does not have a grad_fn",
    ):
        fake_mnist_trainer.fit(max_epochs=1)

    @fake_mnist_trainer.train()
    def _(images, labels):
        return [
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(2.0, requires_grad=False),
            1,
            "some_other_thing",
        ]

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_train_decorator_output_dict(
    fake_mnist_trainer: tao.Trainer, tracker: tao.Tracker
):
    @fake_mnist_trainer.train()
    def _(images, labels):
        return {
            "cls_loss": torch.tensor(1.0, requires_grad=True),
            "seg_loss": torch.tensor(2.0, requires_grad=True),
        }

    with pytest.raises(
        KeyError,
        match="loss",
    ):
        fake_mnist_trainer.fit(max_epochs=1)

    @fake_mnist_trainer.train()
    def _(images, labels):
        return {
            "loss": torch.tensor(3.0, requires_grad=True),
            "cls_loss": torch.tensor(1.0, requires_grad=False),
            "seg_loss": torch.tensor(2.0, requires_grad=True),
        }

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_eval_decorator(fake_mnist_trainer: tao.Trainer):
    @fake_mnist_trainer.eval()
    def _(images, labels):
        assert not fake_mnist_trainer.model.training

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_eval_should_not_have_grad(fake_mnist_trainer: tao.Trainer):
    @fake_mnist_trainer.eval()
    def _(images, labels):
        logits = fake_mnist_trainer.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        with pytest.raises(
            RuntimeError,
            match="element 0 of tensors does not require grad and does not have a grad_fn",
        ):
            loss.backward()

    fake_mnist_trainer.fit(max_epochs=1)


def test_trainer_to_cpu(fake_mnist_trainer: tao.Trainer):
    assert fake_mnist_trainer.device.type == "cpu"
    assert next(fake_mnist_trainer.model.parameters()).device.type == "cpu"
    fake_mnist_trainer.to("cpu")
    assert fake_mnist_trainer.device.type == "cpu"
    assert next(fake_mnist_trainer.model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_trainer_to_cuda(fake_mnist_trainer: tao.Trainer):
    assert fake_mnist_trainer.device.type == "cpu"
    assert next(fake_mnist_trainer.model.parameters()).device.type == "cpu"
    fake_mnist_trainer.to("cuda")
    assert fake_mnist_trainer.device.type == "cuda"
    assert next(fake_mnist_trainer.model.parameters()).device.type == "cuda"
