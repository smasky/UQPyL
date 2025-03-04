import numpy as np
from typing import Union

from .base_kernel import BaseKernel

class Exp(BaseKernel):
    
    def __init__(self, heterogeneous: bool=True,
                 theta: Union[float, np.ndarray]=0.1, 
                 theta_attr: Union[dict, None]= {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}):
        
        super().__init__(heterogeneous, theta, theta_attr)
    
    def __call__(self, D: np.ndarray):
        '''
            Parameters:
                D: np.ndarray
                    The distance matrix
        '''
        theta=self.setting.getVals("theta")
        
        td= -theta
        r= np.exp(np.sum(D*td, axis=1))
        
        return r    