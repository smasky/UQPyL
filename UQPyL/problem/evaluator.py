from .eval import Eval
from .evaluator_base import EvaluatorBase


class Evaluator(EvaluatorBase):
    def __init__(self, objFunc=None, conFunc=None):
        self.objFunc = objFunc
        self.conFunc = conFunc

    def evaluate(self, X, target=None):
        objs = None
        if target in (None, 'objs') and self.objFunc is not None:
            objs = self.objFunc(X)

        cons = None
        if target in (None, 'cons') and self.conFunc is not None:
            cons = self.conFunc(X)

        return Eval(objs=objs, cons=cons, target=target)
