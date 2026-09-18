from .eval import Eval
from .evaluator_base import ModelEvaluatorBase


class ModelEvaluator(ModelEvaluatorBase):
    def __init__(self, objFunc=None, conFunc=None):
        self.objFunc = objFunc
        self.conFunc = conFunc

    def evaluate(self, X, simContext, target=None):
        sims = simContext.sims

        objs = None
        if target in (None, 'objs') and self.objFunc is not None:
            objs = self.objFunc(X, simContext)

        cons = None
        if target in (None, 'cons') and self.conFunc is not None:
            cons = self.conFunc(X, simContext)

        return Eval(objs=objs, cons=cons, sims=sims, target=target)
