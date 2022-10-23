import logging

from ignite.engine import Engine, Events

from ignite.metrics import Metric as IMetric

import pytorch_tao as tao

from pytorch_tao.plugins.base import ValPlugin

logger = logging.getLogger(__name__)


class Metric(ValPlugin):
    """Adapter plugin for ignite metrics.
    Use this plugin as a adapter for using :doc:`ignite:metrics`.

    Args:
        name: name of the metric score
        metric: :class:`ignite:ignite.metrics.metric.Metric` instance
        tune: if tune based on this metric

    .. code-block:: python

        import pytorch_tao as tao
        from pytorch_tao.plugin import Metric
        from ignite.metrics import Accuracy

        trainer = tao.Trainer()
        trainer.use(Metric("accuracy", Accuracy()))
    """

    __skip_tao_event__ = ["_metric"]

    def __init__(self, name: str, metric: IMetric, tune=False):
        self._metric = metric
        self.name = name
        self.tune = tune

    def attach(self, engine: Engine):
        self._metric.attach(engine, self.name)
        super().attach(engine)

    @tao.on(Events.EPOCH_COMPLETED)
    def _track(self, engine: Engine):
        logger.info(f"metric {self.name}: {engine.state.metrics[self.name]}")
        tao.tracker.add_points({self.name: engine.state.metrics[self.name]})
        if self.tune and tao.trial is not None:
            tao.trial.report(engine.state.metrics[self.name], engine.state.epoch)

    @tao.on(Events.COMPLETED)
    def _tell(self, engine: Engine):
        if self.tune and tao.study is not None:
            tao.tell(engine.state.metrics[self.name])
