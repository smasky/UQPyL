from .problemABC import ProblemABC
import numpy as np
from typing import Union, Optional

class Problem(ProblemABC):
    """
    Concrete implementation of the ProblemABC class for defining specific optimization problems.
    
    This class allows users to define custom objective and constraint functions for optimization
    problems, extending the abstract base class ProblemABC.
    
    Methods:
        __init__: Initialize the problem with input/output dimensions, bounds, and custom functions.
    """
    
    def __init__(self, nInput: int, nOutput: int, 
                 ub: Union[int, float, np.ndarray, list], lb: Union[int, float, np.ndarray, list], 
                 objFunc: Optional[callable] = None, conFunc: Optional[callable] = None, 
                 evaluate: Optional[callable] = None,
                 conWgt: Optional[list] = None,
                 varType: list = None, varSet: list = None, optType: Union[list, str] = 'min',
                 xLabels: list = None, yLabels: list = None, name: str = None):
        """
        Initialize the problem with input/output dimensions, bounds, and custom functions.
        
        :param nInput: Number of input variables.
        :param nOutput: Number of output variables.
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
        :param yLabels: Labels for output variables.
        :param name: Name of the problem.
        """
        
        self.objFunc_ = None
        self.conFunc_ = None
        self.evaluate_ = None
        
        if objFunc:
            self.objFunc_ = objFunc
        
        if conFunc:
            self.conFunc_ = conFunc
        
        if evaluate:
            self.evaluate_ = evaluate
        
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        
        super().__init__(nInput=nInput, nOutput=nOutput, ub=ub, lb=lb,
                         conWgt=conWgt, varType=varType, varSet=varSet, 
                         xLabels=xLabels, yLabels=yLabels, optType=optType)