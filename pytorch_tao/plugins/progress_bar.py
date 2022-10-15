import logging
import warnings
from datetime import datetime

import torch

try:
    import psutil
except ModuleNotFoundError:
    psutil = None
try:
    import pynvml
except ModuleNotFoundError:
    pynvml = None
from typing import Tuple

from ignite.engine import Engine, Events
from tqdm import tqdm

import pytorch_tao as tao
from pytorch_tao import helper
from pytorch_tao.plugins import BasePlugin


logger = logging.getLogger(__name__)


class ProgressBar(BasePlugin):
    def __init__(self, *fields: Tuple[str], hardware: bool = True, interval: int = 1):
        super().__init__()
        self.fields = fields
        self.pbar = None
        self.interval = interval
        self.hardware = hardware
        if self.hardware and (psutil is None or pynvml is None):
            warnings.warn(
                "install psutil and pynvml to print hardware usage information"
            )
        self.hardware_record = {}
        self.last_update_time = 0

    def _update_hardware_usage(self):
        now = datetime.now().timestamp()
        if now - self.last_update_time < 1:
            return
        if psutil is not None:
            self.hardware_record["cpu_util"] = psutil.cpu_percent()
            self.hardware_record["memory"] = psutil.virtual_memory()
        if pynvml is not None and torch.cuda.is_available():
            self.hardware_record["gpu_util"] = torch.cuda.utilization()
            self.hardware_record["gpu_memory"] = torch.cuda.mem_get_info()
        self.last_update_time = now

    @tao.on(Events.EPOCH_STARTED)
    def _create_tqdm(self, engine: Engine):
        tpl = "%11s   "
        var = [f"🧪 epoch {engine.state.epoch}"]
        if psutil is not None:
            tpl += "%5s   %12s   "
            var += ["cpu", "memory"]
        if pynvml is not None and torch.cuda.is_available():
            tpl += "%5s   %12s   "
            var += ["gpu", "gpu_memory"]
        tpl += "%8s" * len(self.fields)
        var += self.fields
        print(tpl % tuple(var))
        self.pbar = tqdm(
            total=engine.state.epoch_length,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )

    @tao.on(lambda self: Events.ITERATION_COMPLETED(every=self.interval))
    def _log(self, engine: Engine):
        self._update_hardware_usage()
        self._update_pbar(engine)

    def _update_pbar(self, engine: Engine):
        values = []
        for field in self.fields:
            if field not in engine.state.output:
                logger.warning(
                    f"{field} is not in engines output keys {engine.state.output.keys()} "
                )
            if not helper.is_scalar(engine.state.output[field]):
                raise ValueError(
                    f"ProgressBar only support log scalar field, the field {field} has type {type(field)}"
                )

            values.append(round(helper.item(engine.state.output[field]), 5))
        tpl = "%12s   "
        var = [""]
        if psutil is not None:
            cpu_util = self.hardware_record.get("cpu_util", None)
            memory = self.hardware_record.get("memory", None)
            tpl += "%5s   %12s   "
            var += [
                f"{cpu_util}%" if cpu_util is not None else "N/A",
                f"{round(memory.used / 1e9, 1)}/{round(memory.total / 1e9, 1)} GB"
                if memory is not None
                else "N/A",
            ]
        if pynvml is not None and torch.cuda.is_available():
            gpu_util = self.hardware_record.get("gpu_util", None)
            gpu_memory = self.hardware_record.get("gpu_memory", None)
            tpl += "%5s   %12s   "
            var += [
                f"{gpu_util}%" if gpu_util is not None else "N/A",
                f"{round((gpu_memory[1] - gpu_memory[0])/(1e9), 1)}/{round(gpu_memory[1]/(1e9), 1)} GB"
                if gpu_memory is not None
                else "N/A",
            ]

        tpl += "%8s" * len(self.fields)
        var += values

        self.pbar.set_description(tpl % tuple(var))
        n = engine.state.iteration % engine.state.epoch_length
        self.pbar.n = n if n != 0 else engine.state.epoch_length
        self.pbar.refresh()
        if n == 0:
            self.pbar.close()
