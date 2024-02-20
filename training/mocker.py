class DummySummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def close(self):
        pass


class DummyLRScheduler:
    def step(self, *args, **kwargs):
        pass

    def load_state_dict(self, *args, **kwargs):
        pass

    def state_dict(self, *args, **kwargs):
        return {}
