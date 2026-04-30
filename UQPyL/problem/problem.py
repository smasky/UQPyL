from .base import ProblemBase
import numpy as np
from typing import Union, Optional
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
                 evaluate: Optional[callable] = None,
                 conWgt: Optional[list] = None, nCon: int = 0,
                 varType: list = None, varSet: list = None, optType: Union[list, str] = 'min',
                 xLabels: list = None, name: str = None,
                 space: Optional[SpaceBase] = None,
                 objLabels: list = None, conLabels: list = None):
        """
        Initialize the problem with input/output dimensions, bounds, and custom functions.
        
        :param nInput: Number of input variables.
        :param nObj: Number of objective variables.
        :param ub: Upper bounds for input variables.
        :param lb: Lower bounds for input variables.
        :param objFunc: Custom objective function.
        :param conFunc: Custom constraint function.
        :param evaluate: Custom evaluation function.
        :param conWgt: Constraint weights.
        :param varType: Types of variables (0 for continuous, 1 for integer, 2 for discrete).
        :param varSet: Sets of possible values for discrete variables.
        :param optType: Optimization type ('min' or 'max').
        :param xLabels: Labels for input variables.
        :param name: Name of the problem.
        """
        
        self._validate_callable_config(objFunc, conFunc, evaluate)

        self._obj_fn = None
        self._con_fn = None
        self._eval_fn = None
        
        if objFunc:
            self._obj_fn = objFunc
        
        if conFunc:
            self._con_fn = conFunc
        
        if evaluate:
            self._eval_fn = evaluate
        
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        super().__init__(nInput=nInput, nObj=nObj, ub=ub, lb=lb,
                         conWgt=conWgt, varType=varType, varSet=varSet, 
                         xLabels=xLabels, optType=optType, nCon=nCon,
                         space=space,
                         objLabels=objLabels, conLabels=conLabels)

    @staticmethod
    def _validate_callable_config(objFunc, conFunc, evaluate):
        if evaluate is not None and (objFunc is not None or conFunc is not None):
            raise ValueError("`evaluate` cannot be used together with `objFunc` or `conFunc`.")

        if conFunc is not None and objFunc is None:
            raise ValueError("`conFunc` cannot be used without `objFunc`.")

        if objFunc is None and conFunc is None and evaluate is None:
            raise ValueError("`Problem` requires `objFunc`, `objFunc + conFunc`, or `evaluate`.")
