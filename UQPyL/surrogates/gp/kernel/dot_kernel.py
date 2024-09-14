from typing import Optional
import numpy as np

from .base_kernel import BaseKernel

class DotProduct(BaseKernel):
    """
    Dot-Product Kernel
    
    """
    
    def __init__(self, sigma: float=1.0, sigma_ub: float=1e5, sigma_lb=1e-5):
        
        super().__init__()
        
        self.sigma=sigma
        self.sigma_ub=sigma_ub
        self.sigma_lb=sigma_lb
    
    def __call__(self, trainX, trainY: Optional[np.ndarray]=None):
        
        if trainY is None:
            K=np.inner(trainX, trainX) + self.sigma**2
        else:
            K=np.inner(trainX, trainY) + self.sigma**2
        
        return K