import numpy as np
from typing import Union

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
    
class Setting():
    
    def __init__(self, prefix):
        
        self.prefix=prefix
        self.paras={}
        self.paras_ub={}
        self.paras_lb={}
    
    def setPara(self, key, value, lb, ub):
        
        self.paras[key]=value
        self.paras_lb[key]=lb
        self.paras_ub[key]=ub

    def getPara(self, *args):
        
        values=[]
        for arg in args:
            values.append(self.paras[arg])
        
        if len(args)>1:
            return tuple(values)
        else:
            return values[0]