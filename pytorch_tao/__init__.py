__version__ = "0.1.4"

from pytorch_tao.core import ConfigMissingError, ensure_config, load_cfg
from pytorch_tao.repo import DirtyRepoError, Repo

try:
    repo = Repo.find_by_file(".")
    cfg = load_cfg(repo.cfg_path)
except FileNotFoundError:
    repo = None
    cfg = None

args = None

__all__ = [
    "Repo",
    "load_cfg",
    "cfg",
    "ensure_config",
    "ConfigMissingError",
    "args",
    "DirtyRepoError",
]
