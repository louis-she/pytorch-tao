import sys
from unittest.mock import patch

import pytorch_tao as tao


def test_log_reproduce_command(trainer: tao.Trainer, test_repo: tao.Repo):

    sys.argv = ["main.py", "--batch_size", "10", "--enable_swa"]

    @tao.arguments
    class _:
        batch_size: int = tao.arg(default=32)
        enable_swa: bool = tao.arg(default=False)
        model: str = tao.arg(default="resnet34")

    with patch.object(tao.tracker, "update_meta") as update_meta:
        trainer.use(tao.tracker)
        trainer.fit(max_epochs=1)

        assert test_repo.is_dirty()

        update_meta.assert_not_called()

        test_repo.commit_all("test")
        assert not test_repo.is_dirty()
        trainer.fit(max_epochs=1)
        update_meta.assert_called_once_with(
            {
                "reproduce_command": f"tao run --checkout {tao.repo.head_hexsha(True)} "
                f"{sys.argv[0]} {tao.args.get_command()}"
            }
        )
