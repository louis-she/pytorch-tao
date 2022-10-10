import tempfile
from pathlib import Path

import pytorch_tao as tao
from pytorch_tao.plugins.checkpoint import Checkpoint
from pytorch_tao.plugins.metrics import Metric


def test_checkpoint(trainer: tao.Trainer, simplenet, sum_metric):
    with tempfile.TemporaryDirectory() as tmpdir:
        tao.cfg = type("_", (object,), {"log_dir": tmpdir})
        trainer.use(Metric("sum_score", sum_metric))
        trainer.use(Checkpoint("sum_score", {"model": simplenet}))
        trainer.fit(max_epochs=5)
        assert len(list(Path(tmpdir).glob("*.pt"))) == 3
