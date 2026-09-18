import numpy as np
from typing import Union, Optional

from .base_kernel import BaseKernel

class Cubic(BaseKernel):
    name = "Cubic"
    
    def __init__(self, heterogeneous: bool=True, 
                 theta: Union[float, np.ndarray]=0.1, 
                 theta_attr: Union[dict, None]= {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}
                 ):
        
        super().__init__(heterogeneous, theta, theta_attr)
        
    def __call__(self, D: np.ndarray):
        '''
            Parameters:
                D: np.ndarray
                    The distance matrix
        '''
        nInput = self._checkFeatureMatrix(D, "D")
        self.validateParameters(nInput)
        theta=self.setting.get("theta")
            
        td = np.minimum(1.0, np.abs(D) * theta)
        # DACE correlation is the product of the per-coordinate factors.
        r = np.prod((1.0 - td)**2 * (1.0 + 2.0 * td), axis=1)
        
        return r
