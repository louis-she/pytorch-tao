from typing import Callable


all_events = [
    "started",
    "epoch_started",
    "get_batch_started",
    "get_batch_completed",
    "iteration_started",
    "iteration_completed",
    "dataloader_stop_iteration",
    "exception_raised",
    "terminate_single_epoch",
    "terminate",
    "epoch_completed",
    "completed",
]


def on(event: Callable):
    def decorator(func):
        func._tao_event = event
        return func

    return decorator
