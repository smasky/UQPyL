from .parameter_store import ParameterStore


class Params(ParameterStore):
    """
    Lightweight shared parameter container for runtime-oriented modules.
    """

    def __init__(self):
        super().__init__()

    def set(self, key, value):
        self.data[key] = value

    def get(self, *args):
        values = [self.data[arg] for arg in args]
        if len(args) > 1:
            return tuple(values)
        return values[0]
