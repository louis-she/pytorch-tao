from functools import wraps
import os
from pathlib import Path
from typing import Callable, Dict, List

import yaml

import pytorch_tao as tao


class ConfigMissingError(Exception):

    def __init__(self, keys: List[str], func: Callable):
        self.missing_keys = set(keys)
        self.func = func
        super().__init__(f"Config keys {self.missing_keys} must be present for calling {func.__name__}")


def load_cfg(cfg_path: Path) -> Dict:
    """Load global config, all the config can be accessed by tao.cfg

    By default, tao will load default cfg, if TAO_ENV is set, tao will load the
    specific part of config. TAO_ENV can be used to seperate different running
    environment, like local machine and colab... All the other part of config will
    inherit from default.
    """
    config_name = os.getenv("TAO_ENV", "default")
    with cfg_path.open("r") as f:
        config = yaml.load(f, Loader=yaml.Loader) or {}
    default_config = config.get("default", {})
    env_config = config.get(config_name, {})
    default_config.update(env_config)
    tao.cfg = default_config


def ensure_config(*keys):
    """A decorator to ensure that some config keys must be present for a function
    """
    def decorator(func):
        @wraps(func)
        def real(*args, **kwargs):
            missing_keys = [key for key in keys if key not in tao.cfg]
            if len(missing_keys) != 0:
                raise ConfigMissingError(missing_keys, func)
            return func(*args, **kwargs)
        return real
    return decorator
