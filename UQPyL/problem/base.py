import abc
import functools
import numpy as np
from typing import Union, Optional

from .decorators import singleFunc
from .eval import Eval
from .space import Space, SpaceBase


class ProblemBase(metaclass=abc.ABCMeta):
    """
    Base class for static optimization / analysis problems.
    """

    _evalTargets = (None, "objs", "cons")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        evaluateMethod = cls.__dict__.get("evaluate")
        if evaluateMethod is None or getattr(evaluateMethod, "_uqpylWrapped", False):
            return

        @functools.wraps(evaluateMethod)
        def wrappedEvaluate(self, X, target=None):
            if target not in self._evalTargets:
                raise ValueError(f"The target must be one of {self._evalTargets}.")
            x2d = self.validate(X)
            evalRes = evaluateMethod(self, x2d, target=target)
            return self._validate_eval_result(evalRes, x2d, target)

        wrappedEvaluate._uqpylWrapped = True
        setattr(cls, "evaluate", wrappedEvaluate)

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
            conWgt = np.asarray(conWgt, dtype=float)
            if conWgt.ndim != 1 or conWgt.size != self.nCon:
                raise ValueError("conWgt length must match nCon.")
            if not np.all(np.isfinite(conWgt)) or np.any(conWgt < 0):
                raise ValueError("conWgt must contain finite nonnegative weights.")
            conWgt = conWgt.reshape(1, -1).copy()

        self.conWgt = conWgt
    
    @abc.abstractmethod
    def evaluate(self, X, target=None):
        raise NotImplementedError

    def _validate_common_eval_result(self, evalRes, X):
        if not isinstance(evalRes, Eval):
            raise TypeError("evaluate must return Eval.")

        nSamples = X.shape[0]
        objs = self._coerce_eval_block(evalRes.objs, nSamples, self.nObj, "objs")
        cons = self._coerce_eval_block(evalRes.cons, nSamples, self.nCon, "cons")

        return objs, cons

    @abc.abstractmethod
    def _validate_eval_result(self, evalRes, X, target):
        raise NotImplementedError

    def _coerce_eval_block(self, value, nSamples, nCols, label):
        if value is None:
            return None

        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"`{label}` must be numeric.")

        if arr.ndim == 0:
            if nSamples != 1 or nCols != 1:
                raise ValueError(f"`{label}` scalar output only supports a single sample and single column.")
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            if arr.shape[0] != nSamples:
                raise ValueError(f"`{label}` first dimension must equal n_samples.")
            if nCols != 1:
                raise ValueError(f"`{label}` must be a 2D array with {nCols} columns.")
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            pass
        else:
            raise ValueError(f"`{label}` must be a 1D or 2D numeric array.")

        if arr.shape[0] != nSamples:
            raise ValueError(f"`{label}` first dimension must equal n_samples.")
        if arr.shape[1] != nCols:
            raise ValueError(f"`{label}` second dimension must equal {nCols}.")

        return arr
        
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

    def space_to_unit(self, X):
        return self.space.space_to_unit(X)

    def canonicalize_unit(self, X):
        return self.space.canonicalize_unit(X)

    def apply_var_type(self, X, IFlag=True, DFlag=True):
        return self.space.apply_var_type(X, IFlag=IFlag, DFlag=DFlag)

    def cast_int_vars(self, X):
        return self.space.cast_int_vars(X)

    def map_discrete_vars(self, X):
        return self.space.map_discrete_vars(X)

    singleFunc = staticmethod(singleFunc)
