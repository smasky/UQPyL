import abc
import numpy as np
from typing import Union, Optional, Literal

class ProblemABC(metaclass=abc.ABCMeta):
    """
    Abstract base class for defining optimization problems.
    """

    def __init__(self, nInput:int, nOutput:int,
                 ub: Union[int, float, list, np.ndarray], lb: Union[int, float, list, np.ndarray],
                 nConstraints: int = 0,
                 optType: Union[str, list] = 'min', conWgt: Optional[list] = None,
                 varType: Optional[list] = None, varSet: Optional[dict] = None,
                 xLabels: Optional[list] = None, yLabels: Optional[list] = None):
        """
        Initialize the problem with input and output dimensions, bounds, and other configurations.
        
        :param nInput: Number of input variables.
        :param nOutput: Number of output variables.
        :param ub: Upper bounds for input variables.
        :param lb: Lower bounds for input variables.
        :param optType: Optimization type ('min' or 'max').
        :param conWgt: Constraint weights.
        :param varType: Types of variables (0 for continuous, 1 for integer, 2 for discrete).
        :param varSet: Sets of possible values for discrete variables.
        :param xLabels: Labels for input variables.
        :param yLabels: Labels for output variables.
        """
        
        self.nInput = nInput
        self.nOutput = nOutput
        self.nConstraints = nConstraints
        
        # Set upper and lower bounds
        self._set_ub_lb(ub, lb)
        
        # Check and set optimization type
        self.optType = self._check_optType(optType)
        
        self.encoding = "real"  # Default encoding type
        
        # Set variable types and indices
        if varType is None:
            self.varType = np.zeros(self.nInput)
            self.idxF = np.arange(self.nInput)
            self.idxI = np.array([])
            self.idxD = np.array([])
        else:
            if len(varType) != nInput:
                raise ValueError("The length of varType is not equal to nInput.")
            self.encoding = "mix"
            self.varType = np.array(varType, dtype=np.int32)
            self.idxF = np.where(self.varType == 0)[0]
            self.idxI = np.where(self.varType == 1)[0]
            self.idxD = np.where(self.varType == 2)[0]
        
        # Set variable sets for discrete variables
        if varSet is None:
            self.varSet = {}
        else:
            self.varSet = {}
            for i in self.idxD:
                if isinstance(varSet[i], list):
                    self.varSet[i] = varSet[i]
                else:
                    raise ValueError("The type of sub varSet must be list.")
        
        # Set labels for input and output variables
        if xLabels is None:
            self.xLabels = ['x_' + str(i) for i in range(1, nInput + 1)]
        else:
            self.xLabels = xLabels

        if yLabels is None:
            self.yLabels = ['y_' + str(i) for i in range(1, nOutput + 1)]
        else:
            self.yLabels = yLabels
        
        # Set constraint weights
        if conWgt is not None:
            if not isinstance(conWgt, list):
                raise ValueError('The type of conWgt must be list or None.')
            conWgt = np.array(conWgt).reshape(1, -1)

        self.conWgt = conWgt
        
    def evaluate(self, X):
        """
        Evaluate the problem using either a user-defined or default method.
        
        :param X: Input data to evaluate.
        :return: Dictionary with objectives and constraints.
        """
        
        # Use the user-defined evaluation method if available
        if hasattr(self, 'evaluate_') and self.evaluate_ is not None:
            return self.evaluate_(X)
        
        # Use the default evaluation method
        objs = self.objFunc(X)  # Calculate objectives
        cons = self.conFunc(X)  # Calculate constraints
        
        return {'objs': objs, 'cons': cons}

    def objFunc(self, X):
        """
        Default objective function.
        
        :param X: Input data.
        :return: Array of objective values.
        """
        
        if hasattr(self, 'objFunc_') and self.objFunc_ is not None:
            return self.objFunc_(X)
        
        return np.full((X.shape[0], 1), np.inf)  # Default to infinity if not overridden
    
    def conFunc(self, X):
        """
        Default constraint function.
        
        :param X: Input data.
        :return: Array of constraint values or None.
        """
        
        if hasattr(self, 'conFunc_') and self.conFunc_ is not None:
            return self.conFunc_(X)
        
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
            if len(t) != self.nOutput:
                raise ValueError("The length of optType must be equal to nOutput.")
            
            for i in t:
                if i not in ['min', 'max']:
                    raise ValueError("The optType must be 'min' or 'max'.")
            
            t = [i.lower() for i in t]
            self.opt = np.array([1 if i == 'min' else -1 for i in t])
        else:
            raise ValueError("The type of optType must be str or list.")
        
        return " ".join(t)

    def _transform_discrete_var(self, X):
        """
        Transform discrete variables to their respective sets.
        
        :param X: Input data.
        :return: Transformed input data.
        """
        
        if self.idxD.size != 0:
            for i in self.idxD:
                S = self.varSet[i]
                num_interval = len(S)
                bins = np.linspace(self.lb[0, i], self.ub[0, i], num_interval+1)
                indices = np.digitize(X[:, i], bins, right=False) - 1
                indices[indices == num_interval] = num_interval-1
                X[:, i] = np.array([S[i] for i in indices])
        
        return X
    
    def _transform_int_var(self, X):
        """
        Round integer variables to the nearest integer.
        
        :param X: Input data.
        :return: Transformed input data.
        """
        
        if self.idxI.size != 0:
            X[:, self.idxI] = np.round(X[:, self.idxI])

        return X
    
    def _transform_unit_X(self, X, IFlag=True, DFlag=True):
        """
        Scale and transform input data to the problem's bounds.
        
        :param X: Input data.
        :param IFlag: Flag to transform integer variables.
        :param DFlag: Flag to transform discrete variables.
        :return: Transformed input data.
        """
        
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        
        X_scaled = (X - X_min) / (X_max - X_min)
        X_scaled = X_scaled * (self.ub - self.lb) + self.lb
        
        if self.encoding == 'mix':
            X_scaled = self._transform_to_I_D(X_scaled, IFlag=IFlag, DFlag=DFlag)
        
        return X_scaled 
    
    def _transform_to_I_D(self, X, IFlag=True, DFlag=True):
        """
        Transform input data to integer and discrete variables.
        
        :param X: Input data.
        :param IFlag: Flag to transform integer variables.
        :param DFlag: Flag to transform discrete variables.
        :return: Transformed input data.
        """
        
        if IFlag:
            X = self._transform_int_var(X)
        
        if DFlag:
            X = self._transform_discrete_var(X)
        
        return X
    
    def _set_ub_lb(self, ub: Union[int, float, list, np.ndarray], 
                        lb: Union[int, float, list, np.ndarray]):
        """
        Set upper and lower bounds for input variables.
        
        :param ub: Upper bounds.
        :param lb: Lower bounds.
        """
        
        if isinstance(ub, (int, float)):
            self.ub = np.ones((1, self.nInput)) * ub
        elif isinstance(ub, np.ndarray):
            self._check_bound(ub)
            self.ub = np.atleast_2d(ub)
        elif isinstance(ub, list):
            self.ub = np.atleast_2d(ub)
            self._check_bound(self.ub)
        else:
            raise ValueError("The type of ub is not supported.")
        
        if isinstance(lb, (int, float)):
            self.lb = np.ones((1, self.nInput)) * lb
        elif isinstance(lb, np.ndarray):
            self._check_bound(lb)
            self.lb = np.atleast_2d(lb)
        elif isinstance(lb, list):
            self.lb = np.atleast_2d(lb)
            self._check_bound(self.lb)
        else:
            raise ValueError("The type of lb is not supported.")
    
    def _check_X_2d(self, X):
        """
        Ensure input data is at least 2D.
        
        :param X: Input data.
        :return: 2D input data.
        """
        
        X = np.atleast_2d(X)
        return X
    
    def _check_bound(self, bound: np.ndarray):
        """
        Check if the bounds are consistent with the number of input variables.
        
        :param bound: Bound to check.
        """
        
        bound = bound.ravel()
        if not bound.shape[0] == self.nInput:
            raise ValueError('The input bound is inconsistent with the nInput of the problem setting')
        
    @staticmethod
    def singleFunc(func):
        """
        Decorator to ensure a function works with 2D input data.
        
        :param func: Function to wrap.
        :return: Wrapped function.
        """
        
        def wrapper(X):
            X = np.atleast_2d(X)
            evals = []
            
            for x in X:
                eval = func(x)
                evals.append(np.atleast_1d(eval))

            return np.vstack(evals)
        
        return wrapper
    
    @staticmethod
    def singleEval(func):
        """
        Decorator to ensure an evaluation function works with 2D input data.
        
        :param func: Evaluation function to wrap.
        :return: Wrapped function.
        """
        
        def wrapper(X):
            X = np.atleast_2d(X)
            
            objs = []
            cons = []
            
            for x in X:
                res = func(x)
                objs.append(np.atleast_1d(res['objs']))
                if 'cons' in res:
                    cons.append(np.atleast_1d(res['cons']))
            
            res = {'objs': np.vstack(objs)}
            
            if len(cons) != 0:
                res['cons'] = np.vstack(cons)
                
            return res
        
        return wrapper 