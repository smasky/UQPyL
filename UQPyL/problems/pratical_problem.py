from .problemABC import ProblemABC as Problem
import numpy as np
from typing import Union

class PracticalProblem(Problem):
    def __init__(self, func: callable,
                 nInput: int, nOutput: int, 
                 ub: Union[int, float, np.ndarray], lb: Union[int, float, np.ndarray], 
                 var_type=None, var_set=None, x_labels=None, y_labels=None, name=None):

        self.func = func
        self.name = name
        super().__init__(nInput, nOutput, ub, lb, var_type, var_set, x_labels, y_labels)
        
    def evaluate(self, X):
        return self.func(X)