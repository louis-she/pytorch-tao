from ignite.engine import Events


class _EventBase:

    def __init__(self, every: int = None, once: int = None):
        self.every = every
        self.once = once

    def __call__(self, func):
        func._tao_event = getattr(Events, self.__class__.__name__.upper())(
            every=self.every, once=self.once
        )
        return func


class started(_EventBase):
    pass


class epoch_started(_EventBase):
    pass


class get_batch_started(_EventBase):
    pass


class get_batch_completed(_EventBase):
    pass


class iteration_started(_EventBase):
    pass


class iteration_completed(_EventBase):
    pass


class dataloader_stop_iteration(_EventBase):
    pass


class exception_raised(_EventBase):
    pass


class terminate_single_epoch(_EventBase):
    pass


class terminate(_EventBase):
    pass


class epoch_completed(_EventBase):
    pass


class completed(_EventBase):
    pass
