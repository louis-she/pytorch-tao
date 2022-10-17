from functools import wraps
from typing import Any, Callable, Iterable, List

import torch

from ignite.engine import Engine, Events

from pytorch_tao import helper

from pytorch_tao.plugins.base import BasePlugin, TrainPlugin, ValPlugin


class Trainer:
    """Trainer is just a warp of two engines: train_engine and val_engine"""

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        model: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        train_loader: Iterable = None,
        val_loader: Iterable = None,
        train_func: Callable = lambda e, b: None,
        val_func: Callable = lambda e, b: None,
        val_stride: int = 1,
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
            Events.EPOCH_COMPLETED(every=val_stride), self._do_eval
        )

    def to(self, device: torch.device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self.model:
            self.model.to(self.device)

    def _do_eval(self):
        self.val_engine.run(self.val_loader)

    def train(  # noqa: C901
        self,
        optimizer: torch.optim.Optimizer = None,
        fields: List[str] = None,
        amp: bool = False,
        grad: bool = True,
        accumulate: int = 1,
        scaler: torch.cuda.amp.GradScaler = None,
    ):
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

    def fit(self, *, max_epochs: int):
        self.train_engine.run(self.train_loader, max_epochs=max_epochs)
