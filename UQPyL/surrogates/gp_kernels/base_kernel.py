import numpy as np
from typing import Union

class Gp_Kernel():
    def __init__(self):
        self._theta={}
        self._theta_ub={}
        self._theta_lb={}
    
    def __check_array__(self, value: Union[float,np.ndarray]):
        
        if isinstance(value, float):
            value=np.array([value])
        elif isinstance(value, np.ndarray):
            if value.ndim>1:
                value=value.ravel()
        else:
            raise ValueError("Please make sure the type of value")
        
        return value  