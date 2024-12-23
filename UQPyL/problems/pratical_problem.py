from .problemABC import ProblemABC as Problem
import numpy as np
from typing import Union, Optional

class PracticalProblem(Problem):
    def __init__(self, objFunc: callable,
                 nInput: int, nOutput: int, 
                 ub: Union[int, float, np.ndarray, list], lb: Union[int, float, np.ndarray, list], 
                 conFunc: Optional[callable]=None,
                 var_type: list=None, var_set: list=None, x_labels: list=None, y_labels: list=None, name: str=None):

        self.objFunc = objFunc
        
        self.conFunc = conFunc
        
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        
        super().__init__(nInput, nOutput, ub, lb, var_type, var_set, x_labels, y_labels)
        
    def evaluate(self, X):
        
        return self.objFunc(X)
    
    def constraint(self, X):
        
        if self.conFunc:
            return self.conFunc(X)
        
        else:
            return super().constraint(X)
    