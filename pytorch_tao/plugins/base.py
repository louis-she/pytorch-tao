from pytorch_tao.plugins import events
from ignite.engine import Engine


class BasePlugin:

    def __init__(self, attach_to: str = None):
        self.attach_to = attach_to

    def attach(self, engine: Engine):
        for key in dir(self):
            func = getattr(self, key)
            tao_event = getattr(func, "_tao_event", None)
            if not tao_event:
                continue
            engine.add_event_handler(tao_event, func)


class TrainPlugin(BasePlugin):

    def __init__(self):
        super().__init__(attach_to="train_engine")


class ValPlugin(BasePlugin):

    def __init__(self):
        super().__init__(attach_to="val_engine")
