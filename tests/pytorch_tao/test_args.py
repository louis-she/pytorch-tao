import sys
from typing import List

import pytest
import pytorch_tao as tao


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