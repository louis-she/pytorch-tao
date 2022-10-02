import functools
from functools import wraps
from typing import Callable, Iterable, Iterator

import torch

from ignite.engine import Engine, Events

from pytorch_tao.plugins.base import BasePlugin, TrainPlugin, ValPlugin


class Trainer:
    """Trainer is just a warp of two engines: train_engine and val_engine"""

    def __init__(
        self,
        device: int = None,
        model: torch.nn.Module = None,
        train_loader: Iterable = None,
        val_loader: Iterable = None,
        train_func: Callable = None,
        val_func: Callable = None,
        val_stride: int = 1,
    ):
        self.device = device
        self.model = model
        self.train_engine = Engine(train_func)
        self.train_loader = train_loader
        if val_func:
            self.val_engine = Engine(val_func)
            self.val_loader = val_loader
        self.train_engine.add_event_handler(
            Events.EPOCH_COMPLETED(every=val_stride), self._do_eval
        )

    def _do_eval(self):
        self.val_engine.run(self.val_loader)

    def forward(self, mode: str, amp=False, fields=None):
        if mode not in ["train", "eval"]:
            raise ValueError("mode should be train or eval")

        def decorator(func: Callable):
            @torch.autocast(self.device, enabled=amp)
            @wraps(func)
            def real(batch):
                getattr(self.model, mode)()
                if isinstance(batch, tuple):
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

            return real

        return decorator

    train = functools.partial(forward, mode="train")
    eval = functools.partial(forward, mode="eval")

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

    def fit(
        self, train_loader: Iterator, val_loader: Iterator = None, *, max_epochs: int
    ):
        self.val_loader = val_loader
        self.train_engine.run(train_loader, max_epochs=max_epochs)
