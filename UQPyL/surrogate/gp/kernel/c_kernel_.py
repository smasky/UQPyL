from typing import Optional
import numpy as np

from .base_kernel import BaseKernel

class Constant(BaseKernel):
    
    """
    Constant
    """
    
    def __init__(self, c: float=1.0, 
                    c_attr: dict = {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}):
        
        super().__init__()
        
        self.setting.setPara('constant', c, c_attr)
        
    def __call__(self, trainX: np.ndarray, trainY: Optional[np.ndarray]=None):
        
        c = self.setting.getVals('constant')
        
        if trainY is None:
            K=np.ones((trainX.shape[0], trainY.shape[0]))*c
        else:
            K=np.ones((trainX.shape[0], trainX.shape[0]))*c
        
        return K