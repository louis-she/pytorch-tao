import os

import pytorch_tao as tao
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.metrics import Accuracy
from pytorch_tao.plugins import Metric, ProgressBar
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@tao.arguments
class _:
    max_epochs: int = tao.arg(default=20)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_loader = DataLoader(
    datasets.MNIST(
        f"{os.getenv('HOME')}/datasets",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    ),
    batch_size=32,
    shuffle=True,
)

val_loader = DataLoader(
    datasets.MNIST(
        f"{os.getenv('HOME')}/datasets",
        train=False,
        transform=transforms.ToTensor(),
    ),
    batch_size=32,
)

model = Net()
optimizer = torch.optim.Adam(lr=3e-4, params=model.parameters())

trainer = tao.Trainer(
    device=device,
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
)


@trainer.train()
def _train(images, targets):
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    return {"loss": loss}


@trainer.eval()
def _val(images, targets):
    logits = model(images)
    return {"y_pred": logits, "y": targets}


trainer.use(ProgressBar("loss"), at="train")
trainer.use(Metric("accuracy", Accuracy()))

trainer.fit(max_epochs=tao.args.max_epochs)
