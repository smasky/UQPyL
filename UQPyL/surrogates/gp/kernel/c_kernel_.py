from typing import Optional
import numpy as np

from .base_kernel import BaseKernel

class Constant(BaseKernel):
    """
    Constant
    """
    def __init__(self, c: float=1.0):
        self.c=c
        super().__init__()
    
    def setHyperPara(self, theta: np.ndarray):
        self.c=theta[0]
    
    def __call__(self, trainX: np.ndarray, trainY: Optional[np.ndarray]=None):
    
        if trainY is None:
            K=np.ones((trainX.shape[0],trainY.shape[0]))*self.c
        else:
            K=np.ones((trainX.shape[0], trainX.shape[0]))*self.c
        
        return K