# class krg_kernels():
import numpy as np
from typing import Union

class BaseKernel():
    
    def __init__(self, heterogeneous: bool,
                 theta: Union[float, np.ndarray], theta_lb: Union[float, np.ndarray], theta_ub: Union[float, np.ndarray],
                 ):
        
        self.setting=Setting()
        
        self.heterogeneous=heterogeneous
        
        self.setPara("theta", theta, theta_lb, theta_ub)
        
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