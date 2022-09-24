import os
import pytorch_tao as tao
from pathlib import Path
from typing import Dict
import yaml


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
