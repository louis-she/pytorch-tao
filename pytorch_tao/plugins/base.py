import warnings
from typing import Callable, List

from ignite.engine import CallableEventWithFilter, Engine, EventsList


class BasePlugin:
    """The base class of all the other plugins.

    A plugin is a set of functions that can be hooked into the training or evaluating process.
    It's basicly the same as :doc:`ignite:handlers`.

    Plugin use the same events as :class:`ignite:ignite.engine.events.Events`.

    A Plugin that is inherited directly from BasePlugin should always pass the
    at arguments when calling :code:`trainer.use`, see example bellow.

    Example of create a custom plugin:

    .. code-block:: python

        import pytorch_tao as tao
        from ignite.engine import Events

        class CustomPlugin(BasePlugin):

            def __init__(self):
                self.interval = 3

            @tao.on(Events.ITERATION_COMPLETED)
            def _on_iteration_completed(self, engine: Engine):
                # do something when iteration has completed

            @tao.on(lambda self: Events.ITERATION_COMPLETED(every=self.interval))
            def _on_every_3_iterations_completed(self, engine: Engine):
                # do something when every 3 iterations has completed

        trainer = tao.Trainer()
        trainer.use(CustomPlugin(), at="train")

    In most cases, plugin is phase related, it means that it can be used either for
    training phase or evaluation phase, like :class:`.Scheduler` plugin is only for
    training but :class:`.Metric` plugin is only for evaluation.

    So it's common for a custom plugin to inherited from :class:`.TrainPlugin`
    or :class:`.ValPlugin`.
    """

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
    """Plugin for training phase. For more detail see :class:`BasePlugin`"""

    def __init__(self):
        super().__init__("train")


class ValPlugin(BasePlugin):
    """Plugin for evaluation phase. For more detail see :class:`BasePlugin`"""

    def __init__(self):
        super().__init__("val")
