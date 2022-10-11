import os

import pytorch_tao as tao
import torch
import torch.nn.functional as F
from ignite.metrics import Accuracy
from pytorch_tao.plugins import (
    Checkpoint,
    Metric,
    OutputRecorder,
    ProgressBar,
    Scheduler,
)
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

train_data = datasets.CIFAR10(
    f"{os.getenv('HOME')}/datasets", train=True, transform=transforms.ToTensor()
)
val_data = datasets.CIFAR10(
    f"{os.getenv('HOME')}/datasets", train=False, transform=transforms.ToTensor()
)

trainer = tao.Trainer(
    device="cuda",
    model=model,
    optimizer=optimizer,
    train_loader=DataLoader(train_data, batch_size=128),
    val_loader=DataLoader(val_data, batch_size=128),
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


trainer.use(Scheduler(OneCycleLR(optimizer, 0.1, total_steps=10000)))
trainer.use(Metric("accuracy", Accuracy()))
trainer.use(Checkpoint("accuracy", {"model": model}))
trainer.use(OutputRecorder("loss"), at="train")
trainer.use(ProgressBar("loss"), at="train")

trainer.fit(max_epochs=10)
