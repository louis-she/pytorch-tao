import logging

from ignite.engine import Engine, Events

import pytorch_tao as tao
from pytorch_tao import helper

from pytorch_tao.plugins.base import BasePlugin


logger = logging.getLogger(__name__)


class OutputRecorder(BasePlugin):
    def __init__(self, *fields):
        super().__init__()
        self.fields = fields

    @tao.on(Events.ITERATION_COMPLETED)
    def _record_fields(self, engine: Engine):
        for field in self.fields:
            if field not in engine.state.output:
                logger.warning(
                    f"{field} is not in engines output keys {engine.state.output.keys()} "
                )
            if helper.is_scalar(engine.state.output[field]):
                tao.tracker.add_points({field: helper.item(engine.state.output[field])})
                continue
            # TODO: add image recorder
