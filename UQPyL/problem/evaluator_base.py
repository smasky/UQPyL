import abc


class EvaluatorBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, X, target=None):
        raise NotImplementedError


class ModelEvaluatorBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, X, simContext, target=None):
        raise NotImplementedError


def coerce_evaluator(evaluator):
    if evaluator is None:
        return None
    if isinstance(evaluator, EvaluatorBase):
        return evaluator
    raise TypeError("`evaluator` must be an EvaluatorBase instance.")


def coerce_model_evaluator(evaluator):
    if evaluator is None:
        return None
    if isinstance(evaluator, ModelEvaluatorBase):
        return evaluator
    raise TypeError("`evaluator` must be a ModelEvaluatorBase instance.")
