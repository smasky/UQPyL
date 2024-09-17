import numpy as np
from typing import Union, Optional

from .base_kernel import BaseKernel

class Cubic(BaseKernel):
    
    def __init__(self, heterogeneous: bool=True, 
                 theta: Union[float, np.ndarray]=0.1, 
                 theta_lb: Union[float, np.ndarray]=1e-20, theta_ub: Union[float, np.ndarray]=1
                 ):
        
        super().__init__(heterogeneous, theta, theta_lb, theta_ub)
        
    def __call__(self, D: np.ndarray):
        '''
            Parameters:
                D: np.ndarray
                    The distance matrix
        '''
        theta=self.getPara("theta")
            
        td=np.sum(D*theta, axis=1)
        ones=np.ones(td.shape)
        td=np.minimum(ones, td)
        r=1-3*td**2+2*td**3
        
        return r