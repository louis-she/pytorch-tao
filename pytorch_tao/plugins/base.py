import warnings
from typing import Callable, List

from ignite.engine import CallableEventWithFilter, Engine, EventsList


class BasePlugin:
    engine: Engine
    __skip_tao_event__: List[str] = []

    def __init__(self, attach_to: str = None):
        self.attach_to = attach_to
        self.engine = None

    def set_engine(self, engine: Engine):
        self.engine = engine

    def attach(self, engine: Engine):
        self.set_engine(engine)
        for key in dir(self):
            if key in self.__skip_tao_event__:
                continue
            func = getattr(self, key)
            tao_event_handler = getattr(func, "_tao_event", None)
            if tao_event_handler is None:
                continue
            if not self._is_event_handler(tao_event_handler):
                try:
                    tao_event_handler = tao_event_handler(self)
                except TypeError:
                    warnings.warn("event handler lambda should return valid event type")
                    continue
                if not self._is_event_handler(tao_event_handler):
                    warnings.warn(
                        f"event handler lambda should return valid event type {tao_event_handler}"
                    )
                    continue
            engine.add_event_handler(tao_event_handler, func)

    def _is_event_handler(self, event_handler: Callable):
        return isinstance(event_handler, EventsList) or isinstance(
            event_handler, CallableEventWithFilter
        )


class TrainPlugin(BasePlugin):
    def __init__(self):
        super().__init__("train")


class ValPlugin(BasePlugin):
    def __init__(self):
        super().__init__("val")
