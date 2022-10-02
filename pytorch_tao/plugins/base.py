from ignite.engine import CallableEventWithFilter, Engine, EventsList


class BasePlugin:
    engine: Engine

    def __init__(self, attach_to: str = None):
        self.engine = None

    def set_engine(self, engine: Engine):
        self.engine = engine

    def attach(self, engine: Engine):
        self.set_engine(engine)
        for key in dir(self):
            func = getattr(self, key)
            tao_event = getattr(func, "_tao_event", None)
            if not (
                isinstance(tao_event, EventsList)
                or isinstance(tao_event, CallableEventWithFilter)
            ):
                continue
            engine.add_event_handler(tao_event, func)


class TrainPlugin(BasePlugin):
    pass


class ValPlugin(BasePlugin):
    pass
