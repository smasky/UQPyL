from .base import ProblemBase
import numpy as np
from typing import Union, Optional
from .evaluator import Evaluator
from .evaluator_base import EvaluatorBase, coerce_evaluator
from .eval import Eval
from .space import SpaceBase

class Problem(ProblemBase):
    """
    Concrete implementation of the ProblemBase class for defining specific optimization problems.
    
    This class allows users to define custom objective and constraint functions for optimization
    problems, extending the abstract base class ProblemBase.
    
    Methods:
        __init__: Initialize the problem with input/output dimensions, bounds, and custom functions.
    """
    
    def __init__(self, nInput: int = None, nObj: int = None,
                 ub: Union[int, float, np.ndarray, list] = None, lb: Union[int, float, np.ndarray, list] = None,
                 objFunc: Optional[callable] = None, conFunc: Optional[callable] = None, 
                 conWgt: Optional[list] = None, nCon: int = 0,
                 varType: list = None, varSet: list = None, optType: Union[list, str] = 'min',
                 xLabels: list = None, name: str = None,
                 space: Optional[SpaceBase] = None,
                 objLabels: list = None, conLabels: list = None,
                 evaluator: Optional[EvaluatorBase] = None):
        """
        Initialize the problem with input/output dimensions, bounds, and custom functions.
        
        :param nInput: Number of input variables.
        :param nObj: Number of objective variables.
        :param ub: Upper bounds for input variables.
        :param lb: Lower bounds for input variables.
        :param objFunc: Custom objective function.
        :param conFunc: Custom constraint function.
        :param conWgt: Constraint weights.
        :param varType: Types of variables (0 for continuous, 1 for integer, 2 for discrete).
        :param varSet: Sets of possible values for discrete variables.
        :param optType: Optimization type ('min' or 'max').
        :param xLabels: Labels for input variables.
        :param name: Name of the problem.
        """

        self._obj_fn = None
        self._con_fn = None
        self.evaluator = coerce_evaluator(evaluator)
        callableConfig = self._validate_callable_config(objFunc, conFunc, evaluator)
        if callableConfig is None:
            hasCustomObjMethod, hasCustomConMethod = False, False
        else:
            hasCustomObjMethod, hasCustomConMethod = callableConfig
        
        if objFunc:
            self._obj_fn = objFunc
        elif hasCustomObjMethod:
            self._obj_fn = self.objFunc
        
        if conFunc:
            self._con_fn = conFunc
        elif hasCustomConMethod:
            self._con_fn = self.conFunc

        if self.evaluator is None:
            self.evaluator = Evaluator(objFunc=self._obj_fn, conFunc=self._con_fn)
        
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        super().__init__(nInput=nInput, nObj=nObj, ub=ub, lb=lb,
                         conWgt=conWgt, varType=varType, varSet=varSet, 
                         xLabels=xLabels, optType=optType, nCon=nCon,
                         space=space,
                         objLabels=objLabels, conLabels=conLabels)

    def _validate_callable_config(self, objFunc, conFunc, evaluator):
        hasCustomObjMethod = self.__class__.objFunc is not Problem.objFunc
        hasCustomConMethod = self.__class__.conFunc is not Problem.conFunc

        if conFunc is not None and objFunc is None:
            raise ValueError("`conFunc` cannot be used without `objFunc`.")

        if evaluator is not None and (objFunc is not None or conFunc is not None):
            raise ValueError("`evaluator` cannot be used together with `objFunc` or `conFunc`.")

        if objFunc is None and conFunc is None and evaluator is None and not hasCustomObjMethod:
            raise ValueError("`Problem` requires `objFunc` or `objFunc + conFunc`.")

        return hasCustomObjMethod, hasCustomConMethod

    def evaluate(self, X, target=None):
        return self.evaluator.evaluate(X, target=target)

    def objFunc(self, X):
        obj_fn = getattr(self, "_obj_fn", None)
        if obj_fn is not None:
            return obj_fn(X)

        raise ValueError("`objFunc` is not defined.")

    def conFunc(self, X):
        con_fn = getattr(self, "_con_fn", None)
        if con_fn is not None:
            return con_fn(X)

        return None

    def _validate_eval_result(self, evalRes, X, target):
        objs, cons = self._validate_common_eval_result(evalRes, X)

        if target in (None, "objs") and objs is None:
            raise ValueError("`Problem` evaluate() requires `objs` to be returned.")
        if target in (None, "cons") and self.nCon > 0 and cons is None:
            raise ValueError("`Problem` evaluate() requires `cons` to be returned when nCon > 0.")
        if target == "objs" and cons is not None:
            raise ValueError("`target='objs'` requires `cons` to be None.")
        if target == "cons" and objs is not None:
            raise ValueError("`target='cons'` requires `objs` to be None.")
        if evalRes.sims is not None:
            raise ValueError("`Problem` evaluate() must not return `sims`.")

        return Eval(objs=objs, cons=cons)
