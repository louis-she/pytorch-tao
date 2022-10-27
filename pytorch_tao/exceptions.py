from typing import Callable, List


class DirtyRepoError(Exception):
    """Raised if action need clean repo but it is not"""


class TemplateNotFound(Exception):
    """Raised if jinja template is not found"""


class ConfigMissingError(Exception):
    def __init__(self, keys: List[str], func: Callable):
        self.missing_keys = set(keys)
        self.func = func
        super().__init__(
            f"Config keys {self.missing_keys} must be present for calling {func.__name__}"
        )

