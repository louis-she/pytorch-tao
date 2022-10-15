import os

import pytorch_tao as tao
import torch
import torch.nn.functional as F
from ignite.metrics import Accuracy
from optuna.distributions import CategoricalDistribution, FloatDistribution
from pytorch_tao.plugins import (
    Checkpoint,
    Metric,
    OutputRecorder,
    ProgressBar,
    Scheduler,
)

# from pytorch_tao.trackers import WandbTracker  # Uncomment to use wandb
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class _args:
    max_epochs: int = tao.arg(default=10)
    batch_size: int = tao.arg(
        default=128, tune=CategoricalDistribution([32, 64, 128, 256])
    )
    lr: float = tao.arg(default=128, tune=FloatDistribution(low=3e-4, high=3e-2))


tao.arguments(_args)

# tracker = WandbTracker(tao.name)  # Uncomment to use wandb
# tao.set_tracker(tracker)  # Uncomment to use wandb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=tao.args.lr)

train_loader = DataLoader(
    datasets.CIFAR10(
        f"{os.getenv('HOME')}/datasets", train=True, transform=transforms.ToTensor()
    ),
    batch_size=tao.args.batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    datasets.CIFAR10(
        f"{os.getenv('HOME')}/datasets", train=False, transform=transforms.ToTensor()
    ),
    batch_size=tao.args.batch_size,
)

trainer = tao.Trainer(
    device="cuda",
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
)


@trainer.train(amp=True)
def train(images, targets):
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    return {"loss": loss}


@trainer.eval()
def val_batch(images, targets):
    logits = model(images)
    return {"y_pred": logits, "y": targets}


trainer.use(
    Scheduler(
        OneCycleLR(
            optimizer,
            0.05,
            epochs=tao.args.max_epochs,
            steps_per_epoch=len(train_loader),
        )
    )
)

# trainer.use(tracker)  # Uncomment to use wandb
trainer.use(ProgressBar("loss"), at="train")
trainer.use(Metric("accuracy", Accuracy()))
trainer.use(Checkpoint("accuracy", {"model": model}))
trainer.use(OutputRecorder("loss"), at="train")

trainer.fit(max_epochs=tao.args.max_epochs)
