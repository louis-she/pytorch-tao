Quick Start
===========

0. Installation

.. code-block:: bash

  pip install pytorch_tao

1. Create a new project using :doc:`cli`.

.. code-block:: bash

  tao new tao_project
  cd tao_project

2. open ``main.py``, finish the code by comments. Or replace ``main.py`` with the following code.

.. note::

  Install ``torchvision`` before running the following code.

.. code-block:: python

  import torch
  import pytorch_tao as tao
  from pytorch_tao.plugins import ProgressBar, Metric
  from ignite.metrics import Accuracy

  from torch.utils.data import DataLoader
  from torchvision.datasets import MNIST
  from torchvision.models import resnet18
  from torchvision.transforms import ToTensor

  @tao.arguments
  class _:
      max_epochs: int = tao.arg(default=20)

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  train_loader = DataLoader(MNIST("./", download=True, transform=ToTensor()), batch_size=32)
  val_loader = DataLoader(MNIST("./", train=False, transform=ToTensor()), batch_size=32)

  model = resnet18(num_classes=10)
  model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  optimizer = torch.optim.Adam(model.parameters())

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
      loss = torch.nn.functional.cross_entropy(logits, targets)
      return {"loss": loss}

  @trainer.eval()
  def _eval(images, targets):
      logits = model(images)
      return logits, targets

  trainer.use(ProgressBar("loss"), at="train")
  trainer.use(Metric("accuracy", Accuracy()))
  trainer.fit(max_epochs=tao.args.max_epochs)


This is a MNIST code with only plugins :class:`.ProgressBar` and :class:`.Metric`.For more complicated examples, see https://github.com/louis-she/pytorch-tao/tree/master/examples.

3. Use the following command to train for 10 epochs

.. code-block:: bash

  tao run --dirty main.py --max_epochs 10

.. note::

  Add ``--dirty`` option because that the git repo is dirty, we can omit this option if commit the changes before tao run. It's recommended to use use ``--dirty`` option only for testing purpose.
