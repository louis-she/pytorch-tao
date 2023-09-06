import logging

import ignite.distributed as idist

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

    def __init__(self, name: str, metric: IMetric, tune=False, direction="max"):
        self._metric = metric
        self.name = name
        self.tune = tune
        self.direction = direction
        if self.direction not in ("max", "min"):
            raise ValueError("direction should be max or min")
        if self.direction == "max":
            self.best_score = -float("inf")
        else:
            self.best_score = float("inf")

    def attach(self, engine: Engine):
        self._metric.attach(engine, self.name)
        super().attach(engine)

    @idist.one_rank_only()
    @tao.on(Events.EPOCH_COMPLETED)
    def _track(self, engine: Engine):
        score = engine.state.metrics[self.name]
        logger.info(f"metric {self.name}: {engine.state.metrics[self.name]}")
        tao.tracker.add_points({self.name: engine.state.metrics[self.name]})
        if (self.direction == "max" and score > self.best_score) or (
            self.direction == "min" and score < self.best_score
        ):
            self.best_score = score
        tao.tracker.add_points(
            {self.name: score, f"best_of_{self.name}": self.best_score}
        )
        if self.tune and tao.trial is not None:
            tao.trial.report(engine.state.metrics[self.name], self.trainer.state.epoch)
        # update trainer metrics
        self.trainer.state.metrics = engine.state.metrics
