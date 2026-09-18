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
        
        self._setKernelParameter('constant', c, c_attr)
        
    def __call__(self, trainX: np.ndarray, trainY: Optional[np.ndarray]=None):
        
        self._validateInputs(trainX, trainY)
        c = self.setting.get('constant')
        
        nOther = trainX.shape[0] if trainY is None else trainY.shape[0]
        K = np.full((trainX.shape[0], nOther), c)
        
        return K

    def diag(self, X):
        self._validateInputs(X)
        return np.full(len(X), float(np.asarray(self.setting.get('constant')).item()))
