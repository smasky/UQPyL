from .problemABC import ProblemABC
import numpy as np
from typing import Union, Optional

class Problem(ProblemABC):
    
    def __init__(self, nInput: int, nOutput: int, 
                 ub: Union[int, float, np.ndarray, list], lb: Union[int, float, np.ndarray, list], 
                 objFunc: Optional[callable] = None, conFunc: Optional[callable] = None, 
                 evaluate: Optional[callable] = None,
                 conWgt: Optional[list] = None,
                 varType: list = None, varSet: list = None, optType: Union[list, str] = 'min',
                 xLabels: list = None, yLabels: list = None, name: str = None):
        
        self.objFunc_ = None; self.conFunc_ = None; self.evaluate_ = None
        
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
        
        super().__init__(nInput = nInput, nOutput = nOutput, ub = ub, lb = lb,
                         conWgt = conWgt, varType = varType, varSet = varSet, 
                         xLabels = xLabels, yLabels = yLabels, optType = optType)