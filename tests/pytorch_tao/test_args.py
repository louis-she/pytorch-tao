import sys
from typing import Dict, List, Tuple

import optuna

import pytest
import pytorch_tao as tao
from optuna.distributions import CategoricalDistribution, FloatDistribution


class _Argument:
    a: int
    b: float
    c: str
    d: bool

    e: List[int]
    f: List[float]
    g: List[str]

    h: int = 1
    i: float = 1.1
    j: str = "hello"
    k: bool = True

    l: List[int] = [1, 2, 3]
    m: List[float] = [1.1, 2.2, 3.3]
    n: List[str] = ["hello", "world"]


class _DistributionArgument:
    a: int = 1
    b: int = tao.arg(default=2)
    c: float = tao.arg(default=18.00, tune=FloatDistribution(19.0, 20.0))
    d: str = tao.arg(default="SGD", tune=CategoricalDistribution(["Adam", "AdamW"]))


@pytest.fixture(scope="function")
def empty_argv():
    prev_argv = sys.argv
    sys.argv = ["mock.py"]
    yield
    sys.argv = prev_argv


def test_arguments_default(empty_argv):

    tao.arguments(_Argument)

    assert tao.args.a is None
    assert tao.args.e is None

    assert tao.args.h == 1
    assert tao.args.k is True

    assert tao.args.l[2] == 3
    assert tao.args.m[2] == 3.3
    assert tao.args.n[1] == "world"


def test_arguments_passing(empty_argv):
    command = (
        "mock.py --a 1 --b 1.2 --c hello --d --e 1 2 3 --f 1.1 2.2 3.3 --g hello world"
    )
    sys.argv = command.split(" ")

    tao.arguments(_Argument)

    assert tao.args.a == 1
    assert tao.args.b == 1.2
    assert tao.args.c == "hello"

    assert tao.args.d is True
    assert tao.args.e == [1, 2, 3]
    assert tao.args.f == [1.1, 2.2, 3.3]
    assert tao.args.g == ["hello", "world"]


def test_arguments_prior(empty_argv):
    command = "mock.py --c 17.0"
    sys.argv = command.split(" ")
    tao.study = optuna.create_study()
    tao.arguments(_DistributionArgument)

    assert tao.args.c == 17.0
    assert tao.args.d in ["Adam", "AdamW"]
    assert tao.args.a == 1


def test_arguments_will_create_trial(empty_argv):
    tao.study = optuna.create_study()
    tao.trial = None
    tao.arguments(_DistributionArgument)
    assert isinstance(tao.trial, optuna.Trial)


def test_arguments_with_wrong_type(empty_argv):
    class _WrongTypeArgument_1:
        a: Tuple[int] = 1

    class _WrongTypeArgument_2:
        a: Dict = {"a": "b"}

    with pytest.raises(TypeError):
        tao.arguments(_WrongTypeArgument_1)

    with pytest.raises(TypeError):
        tao.arguments(_WrongTypeArgument_2)
