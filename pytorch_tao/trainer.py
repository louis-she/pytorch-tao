import functools
from functools import wraps
import logging
from typing import Callable, Iterable

import torch

from ignite.engine import Engine, Events

from pytorch_tao.plugins.base import BasePlugin, TrainPlugin, ValPlugin


class Trainer:
    """Trainer is just a warp of two engines: train_engine and val_engine"""

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        model: torch.nn.Module = None,
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
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_func = train_func
        self.val_func = val_func
        self.train_engine = Engine(self.train_func)
        self.val_engine = Engine(self.val_func)
        self.train_engine.add_event_handler(
            Events.EPOCH_COMPLETED(every=val_stride), self._do_eval
        )

    def _do_eval(self):
        self.val_engine.run(self.val_loader)

    def forward(self, mode: str, fields=None, amp=False, grad: bool = True):
        if mode not in ["train", "eval"]:
            raise ValueError("mode should be train or eval")
        if mode == "train" and self.train_func is not None:
            logging.warning("train decorator will override train_func")
        if mode == "eval" and self.val_func is not None:
            logging.warning("eval decorator will override val_func")

        def decorator(func: Callable):
            @torch.autocast(self.device.type, enabled=amp)
            @torch.set_grad_enabled(grad)
            @wraps(func)
            def _process_func(_, batch):
                if self.model is not None:
                    getattr(self.model, mode)()
                if isinstance(batch, (tuple, list)):
                    batch = tuple(
                        [
                            x.to(self.device) if isinstance(x, torch.Tensor) else x
                            for x in batch
                        ]
                    )
                    if fields is None:
                        return func(*batch)
                    return func(*[batch[f] for f in fields])
                elif isinstance(batch, dict):
                    batch = tuple(
                        [
                            x.to(self.device) if isinstance(x, torch.Tensor) else x
                            for x in batch
                        ]
                    )
                    batch = {
                        x: y.to(self.device) if isinstance(y, torch.Tensor) else y
                        for x, y in batch.items()
                    }
                    if fields is None:
                        return func(**batch)
                    return func(**[batch[f] for f in fields])
                else:
                    raise ValueError(
                        f"the type of batch yield is not supported {type(batch)}"
                    )

            if mode == "train":
                self.train_func = _process_func
                self.train_engine._process_function = _process_func
            elif mode == "eval":
                self.val_func = _process_func
                self.val_engine._process_function = _process_func

        return decorator

    train = functools.partialmethod(forward, mode="train")
    eval = functools.partialmethod(forward, mode="eval")

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
