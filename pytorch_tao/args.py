import argparse
import json
import logging
import sys
from typing import _GenericAlias, Any, Dict, List, Type

from optuna.distributions import BaseDistribution

import pytorch_tao as tao


class _Arg:
    """Arg is a hyper parameter.

    A Arg can provided manually by user from command line interface(tao run),
    or by optuna study(tao tune).
    """

    def __init__(self, default: Any, distribution: BaseDistribution):
        self.default = default
        self.distribution = distribution

    def get(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def set_key(self, key: str):
        self.key = key

    def set_type(self, type: Type):
        self.type = type


class _ArgSet:
    """A set of _Args"""

    _args: Dict[str, _Arg]

    def __init__(self):
        self._args = {}

    def __getattr__(self, name: str):
        return self._args[name].get()

    def add_arg(self, arg: _Arg):
        if arg.key in ["add_arg", "get_distribution"]:
            raise ValueError(f"Arg key {arg.key} conflicts with ArgSet func")
        self._args[arg.key] = arg

    def get_distribution(self):
        return {
            k: v.distribution
            for k, v in self._args.items()
            if v.distribution is not None
        }

    def get_json(self) -> str:
        return json.dumps({k: v.get() for k, v in self._args.items()})


def arg(default: Any, tune: BaseDistribution = None):
    return _Arg(default=default, distribution=tune)


def arguments(cls: Type):  # noqa: C901
    """Decorator of class that determine the arguments(hyperparameters).

    There are 3 ways to obtain a argument value, in the prior order, they are:

    1. commmand line interface
    2. optuna distribution
    3. default value

    even when tuning with `tao tune`, we can still pass command line argument
    to override the distribution one.
    """
    argset = _ArgSet()
    parser = argparse.ArgumentParser(description="PyTorch Tao")
    parser.add_argument("--dirty", action="store_true")
    parser.add_argument("--commit", type=str)
    parser.add_argument("--run_at", type=str)

    for key, type in cls.__annotations__.items():
        if type not in [int, float, str, bool, List[int], List[float], List[str]]:
            raise TypeError()
        arg: _Arg = getattr(cls, key, None)
        if not isinstance(arg, _Arg):
            arg = _Arg(default=arg, distribution=None)
        arg.set_key(key)
        arg.set_type(type)
        argset.add_arg(arg)

        parser_kwargs = dict(default=arg.default, type=type)
        if type == bool and arg.default is True:
            logging.warning(f"The value of type boolean key {key} will always be true")
        if type == bool:
            parser_kwargs["action"] = "store_true"
            del parser_kwargs["type"]
        if isinstance(type, _GenericAlias):
            parser_kwargs["type"] = type.__args__[0]
            parser_kwargs["nargs"] = "+"
        parser.add_argument(f"--{key}", **parser_kwargs)

    parser_args = parser.parse_args()
    if tao.study:
        tao.trial = tao.study.ask(fixed_distributions=argset.get_distribution())

    for arg in argset._args.values():
        if f"--{arg.key}" in sys.argv:
            arg.set_value(getattr(parser_args, arg.key))
        elif tao.trial and arg.key in tao.trial.params:
            arg.set_value(tao.trial.params[arg.key])
        else:
            arg.set_value(getattr(parser_args, arg.key, None))

    tao.args = argset
