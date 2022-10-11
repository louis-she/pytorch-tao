import pytorch_tao as tao
import torch.nn.functional as F
from pytorch_tao.plugins import ProgressBar


def test_progress_bar(fake_mnist_trainer: tao.Trainer):
    @fake_mnist_trainer.train()
    def _(images, targets):
        logits = fake_mnist_trainer.model(images)
        loss = F.cross_entropy(logits, targets)
        return {"loss": loss, "constant": 1.0}

    bar = ProgressBar("loss", "constant")
    fake_mnist_trainer.use(bar, at="train")
    fake_mnist_trainer.fit(max_epochs=1)
