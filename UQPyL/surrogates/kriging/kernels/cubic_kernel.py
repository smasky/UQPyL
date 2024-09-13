import numpy as np
from typing import Union, Optional

from .base_kernel import BaseKernel

class Cubic(BaseKernel):
    
    def __init__(self, theta: Union[float, np.ndarray]=1, 
                 theta_lb: Union[float, np.ndarray]=1e-5, theta_ub: Union[float, np.ndarray]=1e5,
                 heterogeneous: bool=True):
        
        super().__init__(theta, theta_lb, theta_ub, heterogeneous)
        
    def __call__(self, D: np.ndarray):
        '''
            Parameters:
                D: np.ndarray
                    The distance matrix
        '''
        theta=self.getPara("theta")
        nSample, _=D.shape
        if self.heterogeneous and isinstance(theta, float):
            theta=np.ones(nSample)*theta
            
        td=np.sum(D*theta, axis=1)
        ones=np.ones(td.shape)
        td=np.minimum(ones, td)
        r=1-3*td**2+2*td**3
        
        return r