import pytorch_tao as tao
import torch.nn.functional as F
from pytorch_tao.plugins import OutputRecorder


def test_output_recorder(fake_mnist_trainer: tao.Trainer, tracker: tao.Tracker):
    @fake_mnist_trainer.train()
    def _(images, labels):
        logits = fake_mnist_trainer.model(images)
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss}

    fake_mnist_trainer.use(OutputRecorder("loss"), at="train")
    fake_mnist_trainer.fit(max_epochs=1)
    assert len(tracker.points) == 20
    assert "loss" in tracker.points[0]
