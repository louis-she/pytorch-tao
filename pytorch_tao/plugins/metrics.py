from ignite.engine import Engine

from ignite.metrics import Metric as IMetric

from pytorch_tao.plugins.base import ValPlugin


class Metric(ValPlugin):
    def __init__(self, name: str, metric: IMetric):
        self._metric = metric
        self.name = name

    def attach(self, engine: Engine):
        super().attach(engine)
        self._metric.attach(engine, self.name)
