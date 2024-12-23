import numpy as np
from typing import Union

from ...setting import Setting
class BaseKernel():
    def __init__(self):
        
        self.setting=Setting()
        
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
    
    def initialize(self, nInput):
        
        if 'l' in self.setting.parasValue:
            length = self.getPara("l")
            lengthUB  = self.setting.parasUb["l"]
            lengthLB = self.setting.parasLb["l"]
            
            if self.heterogeneous:
                if isinstance(length , float):
                    length  = np.ones(nInput)*length 
                elif length .size==1:
                    length  = np.repeat(length , nInput)
                elif length .size!=nInput:
                    raise ValueError("the dimension of length  is not consistent with the number of input")
                
                if isinstance(lengthUB , float):
                    lengthUB  = np.ones(nInput)*lengthUB 
                elif lengthUB .size==1:
                    lengthUB  = np.repeat(lengthUB , nInput)
                elif lengthUB.size!=nInput:
                    raise ValueError("the dimension of lengthUB is not consistent with the number of input")
                
                if isinstance(lengthLB, float):
                    lengthLB = np.ones(nInput)*lengthLB
                elif lengthLB.size==1:
                    lengthLB = np.repeat(lengthLB, nInput)
                elif lengthLB.size!=nInput:
                    raise ValueError("the dimension of lengthLB is not consistent with the number of input")
            
            self.setPara( "l", length.astype(np.float64) , lengthLB.astype(np.float64), lengthUB.astype(np.float64))