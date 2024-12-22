import numpy as np
from typing import Union, Optional

from .base_kernel import BaseKernel

class Guass(BaseKernel):
    
    def __init__(self, heterogeneous: bool=False, 
                 theta: Union[float, np.ndarray]=1, 
                 theta_lb: Union[float, np.ndarray]=1e-20, theta_ub: Union[float, np.ndarray]=100,
                 ):
        
        super().__init__(heterogeneous, theta, theta_lb, theta_ub)
    
    def __call__(self, D: np.ndarray):
        '''
            Parameters:
                D: np.ndarray
                    The distance matrix
        '''
        theta=self.getPara("theta")
            
        td = D * -theta
        r = np.exp(np.sum(D * td, axis=1))
    
        return r    