import numpy as np
from typing import Union

from ...setting import Setting
class BaseKernel():
    
    def __init__(self, heterogeneous: bool,
                 theta: Union[float, np.ndarray], 
                 theta_attr: Union[dict, None] = None,
                 ):
        
        self.setting = Setting()
        
        self.heterogeneous = heterogeneous
        
        self.setting.setPara("theta", theta, theta_attr)
        
    def initialize(self, nInput):
        
        if "theta" in self.setting.parCon:
            theta = self.setting.parCon["theta"]
            
            if self.heterogeneous:
                if isinstance(theta, float):
                    theta = np.ones(nInput)*theta
                elif theta.size==1:
                    theta = np.repeat(theta, nInput)
                elif theta.size!=nInput:
                    raise ValueError("the dimension of theta is not consistent with the number of input")
            
            self.setting.parVal["theta"] = theta
        
        if "theta" in self.setting.parVal:
            theta = self.setting.parVal["theta"]
            theta_ub = self.setting.parUB["theta"]
            theta_lb = self.setting.parLB["theta"]
            
            if self.heterogeneous:
                
                if isinstance(theta_ub, float):
                    theta_ub = np.ones(nInput, dtype=np.float64)*theta_ub
                elif theta_ub.size == 1:
                    theta_ub = np.repeat(theta_ub, nInput)
                elif theta_ub.size != nInput:
                    raise ValueError("the dimension of theta_ub is not consistent with the number of input")
                
                if isinstance(theta_lb, float):
                    theta_lb = np.ones(nInput, dtype=np.float64)*theta_lb
                elif theta_lb.size==1:
                    theta_lb = np.repeat(theta_lb, nInput)
                elif theta_lb.size!=nInput:
                    raise ValueError("the dimension of theta_lb is not consistent with the number of input")
                
                if isinstance(theta, float):
                    theta = np.ones(nInput, dtype=np.float64)*theta
                elif theta.size==1:
                    theta = np.repeat(theta, nInput)
                elif theta.size!=nInput:
                    raise ValueError("the dimension of theta is not consistent with the number of input")
                
                self.setting.parVal["theta"] = theta
                self.setting.parUB["theta"] = theta_ub
                self.setting.parLB["theta"] = theta_lb