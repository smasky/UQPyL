import abc


class SimulatorBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self, X):
        raise NotImplementedError
