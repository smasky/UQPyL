import numpy as np
from typing import Union

from ...setting import Setting
class BaseKernel():
    def __init__(self):
        
        self.setting = Setting()
        
    def __check_array__(self, value: Union[float,np.ndarray]):
        
        if isinstance(value, float):
            value = np.array([value])
        elif isinstance(value, np.ndarray):
            if value.ndim > 1:
                value = value.ravel()
        else:
            raise ValueError("Please make sure the type of value")
        
        return value
    
    def initialize(self, nInput):
        
        if 'l' in self.setting.parCon:
            length = self.setting.parCon["l"]
            
            if self.heterogeneous:
                if isinstance(length , float):
                    length  = np.ones(nInput)*length 
                elif length .size == 1:
                    length  = np.repeat(length , nInput)
                elif length.size != nInput:
                    raise ValueError("the dimension of length  is not consistent with the number of input")

            self.setting.parVal["l"] = length
        
        if 'l' in self.setting.parVal:
            length = self.setting.parVal["l"]
            lengthUB  = self.setting.parUB["l"]
            lengthLB = self.setting.parLB["l"]
            
            if self.heterogeneous:
                if isinstance(length , float):
                    length  = np.ones(nInput)*length 
                elif length .size==1:
                    length  = np.repeat(length , nInput)
                elif length .size!=nInput:
                    raise ValueError("the dimension of length  is not consistent with the number of input")
                
                if isinstance(lengthUB , float):
                    lengthUB  = np.ones(nInput)*lengthUB 
                elif lengthUB.size == 1:
                    lengthUB  = np.repeat(lengthUB , nInput)
                elif lengthUB.size != nInput:
                    raise ValueError("the dimension of lengthUB is not consistent with the number of input")
                
                if isinstance(lengthLB, float):
                    lengthLB = np.ones(nInput)*lengthLB
                elif lengthLB.size == 1:
                    lengthLB = np.repeat(lengthLB, nInput)
                elif lengthLB.size != nInput:
                    raise ValueError("the dimension of lengthLB is not consistent with the number of input")
            
            self.setting.parVal["l"] = length
            self.setting.parUB["l"] = lengthUB
            self.setting.parLB["l"] = lengthLB