from typing import Callable, Iterable
from ignite.engine import Engine, Events
from pytorch_tao.plugins.base import BasePlugin, TrainPlugin, ValPlugin


class Trainer:
    """Trainer is just a warp of two engines: train_engine and val_engine"""

    def __init__(
        self,
        train_func: Callable,
        train_loader: Iterable,
        val_func: Callable = None,
        val_loader: Iterable = None,
        val_stride: int = 1,
        max_epochs: int = 1e9
    ):
        self.train_engine = Engine(train_func)
        self.train_loader = train_loader
        if val_func and val_loader:
            self.val_engine = Engine(val_func)
            self.val_loader = val_loader
        self.max_epochs = max_epochs

        self.train_engine.add_event_handler(
            Events.EPOCH_COMPLETED(every=val_stride),
            self._do_eval
        )

    def _do_eval(self):
        self.val_engine.run(self.val_loader)

    def use(self, plugin: BasePlugin):
        if isinstance(plugin, TrainPlugin):
            plugin.attach(self.train_engine)
        elif isinstance(plugin, ValPlugin):
            if not hasattr(self, "val_engine"):
                raise ValueError("use a val plugin without val_func and val_loader")
            plugin.attach(self.val_engine)
        else:
            raise ValueError("base plugin should maunally attach to engine")

    def start(self):
        self.train_engine.run(self.train_loader, max_epochs=self.max_epochs)
