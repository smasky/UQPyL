class ParameterStore:
    """
    Minimal shared parameter container interface.

    This base only provides map-like access helpers. Rich parameter-space
    semantics such as bounds, categorical encoding, and owner partitioning
    remain in specialized subclasses like surrogate Setting.
    """

    def __init__(self):
        self.data = {}

    def _mapping(self):
        return self.data

    @property
    def dicts(self):
        return self._mapping()

    def keys(self):
        return self._mapping().keys()

    def values(self):
        return self._mapping().values()

    def items(self):
        return self._mapping().items()

    def asDict(self):
        return dict(self._mapping())
