import argparse
from ast import Tuple
from inspect import getmembers
from typing import List, Type, _GenericAlias
import logging
import pytorch_tao as tao


def arguments(cls: Type):
    """Decorator of class that add arguments to parser"""
    parser = argparse.ArgumentParser(description="PyTorch Tao")
    for key, type in cls.__annotations__.items():
        if type not in [int, float, str, bool, List[int], List[float], List[str]]:
            raise TypeError()
        defaultvalue = getattr(cls, key, None)
        parser_kwargs = dict(default=defaultvalue, type=type)
        if type == bool and defaultvalue is True:
            logging.warn(f"The value of type boolean key {key} will always be true")
        if type == bool:
            parser_kwargs["action"] = "store_true"
            del parser_kwargs["type"]
        if isinstance(type, _GenericAlias):
            parser_kwargs["type"] = type.__args__[0]
            parser_kwargs["nargs"] = "+"
        parser.add_argument(f"--{key}", **parser_kwargs)
    tao.args = parser.parse_args()
