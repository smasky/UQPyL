import numpy as np
from typing import Union

from ...setting import Setting
class BaseKernel():
    def __init__(self):
        
        self.setting=Setting("kernel")
        
    def __check_array__(self, value: Union[float,np.ndarray]):
        
        if isinstance(value, float):
            value=np.array([value])
        elif isinstance(value, np.ndarray):
            if value.ndim>1:
                value=value.ravel()
        else:
            raise ValueError("Please make sure the type of value")
        
        return value
        
    def setPara(self, key, value, lb, ub):
        
        self.setting.setPara(key, value, lb, ub)
    
    def getPara(self, *args):
        
        return self.setting.getPara(*args)