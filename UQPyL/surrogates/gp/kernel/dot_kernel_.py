from typing import Optional
import numpy as np

from .base_kernel import BaseKernel

class DotProduct(BaseKernel):
    """
    Dot-Product Kernel
    
    """
    
    def __init__(self, sigma: float=1.0, 
                 sigma_attr: dict = {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}):
        
        super().__init__()
        
        self.setting.setPara('sigma', sigma, sigma_attr)
    
    def __call__(self, trainX, trainY: Optional[np.ndarray]=None):
        
        sigma = self.setting.getVals('sigma')
        
        if trainY is None:
            K=np.inner(trainX, trainX) + sigma**2
        else:
            K=np.inner(trainX, trainY) + sigma**2
        
        return K