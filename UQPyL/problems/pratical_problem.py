from .problemABC import ProblemABC as Problem
import numpy as np
from typing import Union, Optional

class PracticalProblem(Problem):
    
    def __init__(self, nInput: int, nOutput: int, 
                 ub: Union[int, float, np.ndarray, list], lb: Union[int, float, np.ndarray, list], 
                 conFunc: Optional[callable] = None, objFunc: Optional[callable] = None,
                 conWgt: Optional[list] = None,
                 varType: list = None, varSet: list = None, 
                 xLabels: list = None, yLabels: list = None, name: str = None):
        
        self.objFunc = objFunc
        
        if conFunc:
            self.conFunc = conFunc
        
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        
        super().__init__(nInput = nInput, nOutput = nOutput, 
                        ub = ub, lb = lb, objFunc = objFunc, conFunc = conFunc, 
                        conWgt = conWgt, varType = varType, varSet = varSet, xLabels = xLabels, yLabels = yLabels)