import numpy as np
from typing import Union

from ...setting import Setting
class BaseKernel():
    
    def __init__(self, heterogeneous: bool,
                 theta: Union[float, np.ndarray], theta_lb: Union[float, np.ndarray], theta_ub: Union[float, np.ndarray], 
                 ):
        
        self.setting=Setting("kernel")
        
        self.heterogeneous=heterogeneous
        
        self.setPara("theta", theta, theta_lb, theta_ub)
        
    def setPara(self, key, value, lb, ub):
        
        self.setting.setPara(key, value, lb, ub)
    
    def getPara(self, *args):
        
        return self.setting.getPara(*args)
    
    def initialize(self, nInput):
            
            theta=self.getPara("theta")
            theta_ub=self.setting.paras_ub["theta"]
            theta_lb=self.setting.paras_lb["theta"]
            
            if self.heterogeneous:
                if isinstance(theta, float):
                    theta=np.ones(nInput)*theta
                elif theta.size==1:
                    theta=np.repeat(theta, nInput)
                elif theta.size!=nInput:
                    raise ValueError("the dimension of theta is not consistent with the number of input")
                
                if isinstance(theta_ub, float):
                    theta_ub=np.ones(nInput)*theta_ub
                elif theta_ub.size==1:
                    theta_ub=np.repeat(theta_ub, nInput)
                elif theta_ub.size!=nInput:
                    raise ValueError("the dimension of theta_ub is not consistent with the number of input")
                
                if isinstance(theta_lb, float):
                    theta_lb=np.ones(nInput)*theta_lb
                elif theta_lb.size==1:
                    theta_lb=np.repeat(theta_lb, nInput)
                elif theta_lb.size!=nInput:
                    raise ValueError("the dimension of theta_lb is not consistent with the number of input")
            
            self.setPara("theta", theta, theta_lb, theta_ub)