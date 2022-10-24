import logging
from typing import List

from ignite.engine import Engine, Events

import pytorch_tao as tao
from pytorch_tao import helper

from pytorch_tao.plugins.base import BasePlugin


logger = logging.getLogger(__name__)


class OutputRecorder(BasePlugin):
    """Record the any output using tracker.
    Note that this is inherited from BasePlugin so one should
    pass `at` argument to trainer.

    Args:
        fields: the field of output to track

    ..  list-table:: Hooks
        :header-rows: 1

        * - Hook Point
          - Logic
        * - ITERATION_COMPLETED
          - call :meth:`tracker.add_points` with the outputs

    .. code-block:: python

        import pytorch_tao as tao
        from pytorch_tao.plugin import OutputRecorder

        model = ...
        optimizer = ...

        trainer = tao.Trainer()
        @trainer.train()
        def _train(images, targets):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            return {"loss": loss}

        trainer.use(OutputRecorder("loss"), at="train")
    """

    def __init__(self, *fields: List[str]):
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
