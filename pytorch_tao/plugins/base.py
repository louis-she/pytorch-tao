from pytorch_tao.plugins import events


class BasePlugin:

    def __init__(self, attach_to: str = None):
        self.attach_to = attach_to


class TrainPlugin(BasePlugin):

    def __init__(self):
        super().__init__(attach_to="train_engine")


class ValPlugin(BasePlugin):

    def __init__(self):
        super().__init__(attach_to="val_engine")
