from functools import wraps
from typing import Any, Callable, Iterable, List

import torch

from ignite.engine import Engine, Events
from ignite.engine.events import State

from pytorch_tao import helper

from pytorch_tao.plugins.base import BasePlugin, TrainPlugin, ValPlugin


class Trainer:
    """Trainer is basicly a warp of two :class:`ignite:ignite.engine.engine.Engine`: train_engine and val_engine.

    Args:
        device: torch.device, to train with cuda or cpu.
        model: a pytorch model, this is an optional argument but is highly recommanded
            to pass in for some features to work.
        optimizer: a pytorch optimizer which is also optional but highly recommended to pass in.
        train_loader: dataloader for training stage.
        val_loader: dataloader for evaluation stage, if None then will no do evaluation.
        train_func: training forward function, it's more recommanded to use
            :meth:`train` decorator after trainer is initialized.
        val_func: evaluation forward function, it's more recommanded to use
            :meth:`eval` decorator after trainer is initialized.
        val_stride: epochs stride to do evaluation.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        model: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        train_loader: Iterable = None,
        val_loader: Iterable = None,
        train_func: Callable = lambda e, b: None,
        val_func: Callable = lambda e, b: None,
        val_event: Callable = lambda e, event: True,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_func = train_func
        self.val_func = val_func
        self.train_engine = Engine(self.train_func)
        self.val_engine = Engine(self.val_func)
        self.train_engine.add_event_handler(
            Events.EPOCH_COMPLETED(event_filter=val_event), self._do_eval
        )

    def to(self, device: torch.device):
        """Move to device

        Args:
            device: torch.device
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self.model:
            self.model.to(self.device)

    def _do_eval(self):
        self.val_engine.run(self.val_loader)

    @property
    def state(self) -> State:
        return self.train_engine.state

    def train(  # noqa: C901
        self,
        optimizer: torch.optim.Optimizer = None,
        fields: List[str] = None,
        amp: bool = False,
        grad: bool = True,
        accumulate: int = 1,
        scaler: torch.cuda.amp.GradScaler = None,
    ):
        """Decorator that define the training process.

        Args:
            optimizer: torch optimizer, if given, will override the one
                passed in the constructor.
            fields: fields to selecte from the yield batch by dataloader. if None,
                then the all yield data will be passed in as the decorated functions parameters.
                If the yield type is of dict, then will be passed as keywords args, if it's a
                tuple then will be passed as positional args.
            amp: whether to use amp.
            grad: whther to enabled grad.
            accumulate: gradient accumulation step.
            scaler: if amp is enabled, which scaler to use.

        Ideally, the decorated function should do the following:

        0. the parameters should match the fields arguments.

        1. do model forwarding and compute loss, if multiple loss then
           a scalar loss should be made by weighted sum.

        2. return at least a scalar represent the final loss, if returning
           type is dict, then a `loss` key should be there, if returning type
           if tuple, then the first element will be treated as the sum loss.

        See the following code examples for details.

        .. code-block:: python

            import pytorch_tao as tao

            trainer = tao.Trainer()

            # assume that dataloader yields a tuple of (images, targets)
            @trainer.train()
            def _train(images, targets):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                return {"loss": loss}


        .. code-block:: python

            import pytorch_tao as tao

            trainer = tao.Trainer()

            # assume that dataloader yields a tuple of (images, targets, metadata)
            @trainer.train(fields=(0, 1))
            def _train(images, targets):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                return {"loss": loss}

        .. code-block:: python

            import pytorch_tao as tao

            trainer = tao.Trainer()

            # assume that dataloader yields a tuple of (images, targets, metadata)
            @trainer.train()
            def _train(images, targets, metadata):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                return {"loss": loss}
        """
        if amp and self.device.type == "cuda":
            scaler = scaler if scaler is not None else torch.cuda.amp.GradScaler()
        optimizer = optimizer if optimizer is not None else self.optimizer

        def decorator(func: Callable):
            @torch.autocast(self.device.type, enabled=amp)
            @torch.set_grad_enabled(grad)
            @wraps(func)
            def _func(engine: Engine, batch):
                if self.model is not None:
                    self.model.train()
                if (engine.state.iteration - 1) % accumulate == 0:
                    optimizer.zero_grad()
                results = self._process_func(fields, batch, func)
                if helper.is_scalar(results):
                    loss = results
                elif isinstance(results, (list, tuple)):
                    loss = results[0]
                elif isinstance(results, dict):
                    loss = results["loss"]
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if engine.state.iteration % accumulate == 0:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss.backward()
                    if engine.state.iteration % accumulate == 0:
                        optimizer.step()
                return results

            self.train_func = _func
            self.train_engine._process_function = _func

        return decorator

    def eval(
        self,
        fields: List[str] = None,
        amp: bool = False,
    ):
        """Decorator that define the training process.
        Parameters of this has the same functionality as :meth:`train`.

        Args:
            fields: fields to selecte from the yield batch by dataloader. if None,
                then the all yield data will be passed in as the decorated functions parameters.
                If the yield type is of dict, then will be passed as keywords args, if it's a
                tuple then will be passed as positional args.
            amp: whether to use amp.

        Ideally, the decorated function should do the following:

        0. the parameters should match the fields arguments.

        1. do model forwarding and return the raw output of model.

        This function is mainly used for computing metrics.

        Cause different metrics may have different input requirements, like ROC AUC requires
        probabilities but F-score requires hard 0 or 1. So for this method, it should only
        return the raw output from model, and let metrics to do the transformation. Thus
        we can have multiple metrics but only forward the input once.  Luckly that every
        metric from pytorch ignite has a :code:`output_transform` method to do the thing.

        .. code-block:: python

            import pytorch_tao as tao
            from pytorch_tao.plugin import Metric
            from ignite.metrics import Accuracy, ROC_AUC

            trainer = tao.Trainer()

            @trainer.eval()
            def _eval(images, targets):
                return model(images), targets

            trainer.use(Metric("accuracy", Accuracy(lambda: logits, targets: logits > 0, targets)))
            trainer.use(Metric("roc_auc", ROC_AUC(lambda: logits, targets: torch.sigmoid(logits), targets)))
        """

        def decorator(func: Callable):
            @torch.autocast(self.device.type, enabled=amp)
            @torch.set_grad_enabled(False)
            @wraps(func)
            def _func(engine: Engine, batch):
                if self.model is not None:
                    self.model.eval()
                results = self._process_func(fields, batch, func)
                return results

            self.val_func = _func
            self.val_engine._process_function = _func

        return decorator

    def _process_func(self, fields: List[str], batch: Any, func: Callable):
        if isinstance(batch, (tuple, list)):
            batch = tuple(
                [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
            )
            if fields is None:
                return func(*batch)
            return func(*[batch[f] for f in fields])

        elif isinstance(batch, dict):
            batch = {
                x: y.to(self.device) if isinstance(y, torch.Tensor) else y
                for x, y in batch.items()
            }
            if fields is None:
                return func(**batch)
            return func(**{f: batch[f] for f in fields})
        else:
            raise ValueError(f"the type of batch yield is not supported {type(batch)}")

    def use(self, plugin: BasePlugin, at: "str" = None):
        """Use a plugin.

        If use a :class:`pytorch_tao.plugins.TrainPlugin`, then it'll be attached to train_engine
        by default, and :class:`pytorch_tao.plugins.ValPlugin` will attached to val_engine as well.

        If the plugin inherited from :class:`pytorch_tao.plugins.BasePlugin`, then using this
        plugin should pass the :code:`at` argument to either "train" or "val".

        Args:
            plugin: the plugin to use.
            at: which engine to attach.
        """
        if at is not None:
            if at not in ["train", "val"]:
                raise ValueError("at must be train or val")
            plugin.attach(getattr(self, f"{at}_engine"))
            return
        if isinstance(plugin, TrainPlugin):
            plugin.attach(self.train_engine)
        elif isinstance(plugin, ValPlugin):
            if not hasattr(self, "val_engine"):
                raise ValueError("use a val plugin without val_func and val_loader")
            plugin.attach(self.val_engine)
        else:
            raise ValueError("base plugin should maunally attach to engine")
        plugin.trainer = self

    def fit(self, *, max_epochs: int):
        """Start the training and evaluation loop process.

        Args:
            max_epochs: how many epochs to train.
        """
        self.train_engine.run(self.train_loader, max_epochs=max_epochs)
