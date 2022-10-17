from unittest.mock import patch

import optuna
import pytest
import pytorch_tao as tao


def test_tao_tell():
    distributions = {
        "optimizer": optuna.distributions.CategoricalDistribution(["adam", "sgd"]),
        "lr": optuna.distributions.LogUniformDistribution(0.0001, 0.1),
    }

    tao.study = optuna.create_study()
    tao.trial = tao.study.ask(distributions)

    with patch.object(tao.study, "tell") as mock_tell:
        tao.tell(0.9)
        mock_tell.assert_called_with(tao.trial, 0.9)


def test_tao_tell_without_tune():
    with pytest.warns(
        UserWarning, match="`tao.tell` is skipped when tao is not in tune mode"
    ):
        tao.tell(1.0)
