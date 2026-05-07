import abc
import numpy as np
from typing import Union, Optional

from .decorators import singleEval, singleFunc
from .eval import Eval
from .space import Space, SpaceBase


class ProblemBase(metaclass=abc.ABCMeta):
    """
    Base class for static optimization / analysis problems.
    """

    def __init__(self, nInput:int = None, nObj:int = None,
                 ub: Union[int, float, list, np.ndarray] = None, lb: Union[int, float, list, np.ndarray] = None,
                 nCon: int = 0,
                 optType: Union[str, list, None] = 'min', conWgt: Optional[list] = None,
                 varType: Optional[list] = None, varSet: Optional[dict] = None,
                 xLabels: Optional[list] = None,
                 space: Optional[SpaceBase] = None,
                 objLabels: Optional[list] = None, conLabels: Optional[list] = None):
        """
        Initialize the problem with input and output dimensions, bounds, and other configurations.
        
        :param nInput: Number of input variables.
        :param nObj: Number of objective variables.
        :param ub: Upper bounds for input variables.
        :param lb: Lower bounds for input variables.
        :param optType: Optimization type ('min' or 'max').
        :param conWgt: Constraint weights.
        :param varType: Types of variables (0 for continuous, 1 for integer, 2 for discrete).
        :param varSet: Sets of possible values for discrete variables.
        :param xLabels: Labels for input variables.
        """
        
        if nObj is None:
            raise ValueError("nObj must be provided.")

        if space is None:
            self.space = Space(
                nInput=nInput,
                ub=ub,
                lb=lb,
                varType=varType,
                varSet=varSet,
                xLabels=xLabels,
            )
        else:
            self.space = space

        self.nInput = self.space.nInput
        self.nObj = nObj
        self.nCon = nCon
        self.nOutput = nObj
        self.nCons = nCon

        self.ub = self.space.ub if hasattr(self.space, "ub") else None
        self.lb = self.space.lb if hasattr(self.space, "lb") else None

        # Check and set optimization type
        self.optType = self._check_optType(optType) if optType is not None else None

        self.varType = getattr(self.space, "varType", None)
        self.idxF = getattr(self.space, "idxF", np.arange(self.nInput))
        self.idxI = getattr(self.space, "idxI", np.array([]))
        self.idxD = getattr(self.space, "idxD", np.array([]))
        self.varSet = getattr(self.space, "varSet", {})

        if xLabels is None:
            self.xLabels = self.space.xLabels
        else:
            self.xLabels = xLabels

        if objLabels is None:
            self.objLabels = ['obj_' + str(i) for i in range(1, self.nObj + 1)]
        else:
            self.objLabels = objLabels
        self.yLabels = self.objLabels

        if self.nCon == 0:
            self.conLabels = None
        elif conLabels is None:
            self.conLabels = ['con_' + str(i) for i in range(1, self.nCon + 1)]
        else:
            self.conLabels = conLabels

        # Set constraint weights
        if conWgt is not None:
            if not isinstance(conWgt, list):
                raise ValueError('The type of conWgt must be list or None.')
            conWgt = np.array(conWgt).reshape(1, -1)

        self.conWgt = conWgt
    
    def evaluate(self, X, target=None):
        """
        Evaluate the problem using either a user-defined or default method.
        
        :param X: Input data to evaluate.
        :return: Evaluation outputs for objectives and constraints.
        """
        
        # Use the user-defined evaluation method if available
        if target not in (None, "objs", "cons"):
            raise ValueError("The target must be None, 'objs' or 'cons'.")

        eval_fn = getattr(self, "_eval_fn", None)
        if eval_fn is not None:
            eval = eval_fn(X)
            if not isinstance(eval, Eval):
                raise TypeError("evaluate must return Eval.")
            objs = eval.objs
            cons = eval.cons
        else:
            objs = self.objFunc(X) if target in (None, "objs") else None
            cons = self.conFunc(X) if target in (None, "cons") else None
            return Eval(objs=objs, cons=cons)

        if target == "objs":
            cons = None
        elif target == "cons":
            objs = None

        return Eval(objs=objs, cons=cons)

    def objFunc(self, X):
        """
        Default objective function.
        
        :param X: Input data.
        :return: Array of objective values.
        """
        
        obj_fn = getattr(self, "_obj_fn", None)
        if obj_fn is not None:
            return obj_fn(X)
        
        eval_fn = getattr(self, "_eval_fn", None)
        if eval_fn is not None:
            eval = eval_fn(X)
            if not isinstance(eval, Eval):
                raise TypeError("evaluate must return Eval.")
            return eval.objs
        
        return np.full((X.shape[0], 1), np.inf)  # Default to infinity if not overridden
  
    def conFunc(self, X):
        """
        Default constraint function.
        
        :param X: Input data.
        :return: Array of constraint values or None.
        """
        
        con_fn = getattr(self, "_con_fn", None)
        if con_fn is not None:
            return con_fn(X)
        
        eval_fn = getattr(self, "_eval_fn", None)
        if eval_fn is not None:
            eval = eval_fn(X)
            if not isinstance(eval, Eval):
                raise TypeError("evaluate must return Eval.")
            return eval.cons
        
        return None  # Default to None if not overridden
        
    def getOptimum(self):
        """
        Abstract method to get the optimum solution.
        """
        pass
    
    def _check_optType(self, t):
        """
        Validate and set the optimization type.
        
        :param t: Optimization type ('min' or 'max').
        :return: String representation of the optimization type.
        """
        
        if isinstance(t, str):
            if t not in ['min', 'max']:
                raise ValueError("The optType must be 'min' or 'max'.")
            
            self.opt = 1 if t == 'min' else -1
            t = [t.lower()]
        elif isinstance(t, list):
            if len(t) != self.nObj:
                raise ValueError("The length of optType must be equal to nObj.")
            
            for i in t:
                if i not in ['min', 'max']:
                    raise ValueError("The optType must be 'min' or 'max'.")
            
            t = [i.lower() for i in t]
            self.opt = np.array([1 if i == 'min' else -1 for i in t])
        else:
            raise ValueError("The type of optType must be str or list.")
        
        return " ".join(t)

    def validate(self, X):
        return self.space.validate(X)

    def _check_X_2d(self, X):
        return self.validate(X)

    def unit_to_space(self, X, IFlag=True, DFlag=True):
        return self.space.unit_to_space(X, IFlag=IFlag, DFlag=DFlag)

    def apply_var_type(self, X, IFlag=True, DFlag=True):
        return self.space.apply_var_type(X, IFlag=IFlag, DFlag=DFlag)

    def cast_int_vars(self, X):
        return self.space.cast_int_vars(X)

    def map_discrete_vars(self, X):
        return self.space.map_discrete_vars(X)

    # Compatibility wrappers retained during migration.
    def _transform_discrete_var(self, X):
        return self.map_discrete_vars(X)

    def _transform_int_var(self, X):
        return self.cast_int_vars(X)

    def _transform_to_I_D(self, X, IFlag=True, DFlag=True):
        return self.apply_var_type(X, IFlag=IFlag, DFlag=DFlag)

    def _transform_unit_X(self, X, IFlag=True, DFlag=True):
        return self.unit_to_space(X, IFlag=IFlag, DFlag=DFlag)
        
    singleFunc = staticmethod(singleFunc)
    singleEval = staticmethod(singleEval)
