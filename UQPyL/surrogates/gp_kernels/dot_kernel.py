from typing import Optional
import numpy as np

from .base_kernel import Gp_Kernel

class DotProduct(Gp_Kernel):
    """
    Dot-Product Kernel
    
    """
    
    def __init__(self, sigma: float=1.0, sigma_ub: float=1e5, sigma_lb=1e-5):
        
        super().__init__()
        
        self.sigma=sigma
        self.sigma_ub=sigma_ub
        self.sigma_lb=sigma_lb
    
    def __call__(self, trainX, trainY: Optional[np.ndarray]=None):
        
        if trainY is None:
            K=np.inner(trainX, trainX) + self.sigma**2
        else:
            K=np.inner(trainX, trainY) + self.sigma**2
        
        return K
    #################################Attributes##################################
    #--------------------------------sigma--------------------------------------#
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        value=self.__check_array__(value)
        self._sigma=value
        self._theta['sigma']=value
    
    @property
    def sigma_ub(self):
        return self._sigma_ub
    
    @sigma_ub.setter
    def sigma_ub(self, value):
        value=self.__check_array__(value)
        self._sigma_ub=value
        self._theta_ub['sigma']=value
    
    @property
    def sigma_lb(self):
        return self._sigma_lb
    
    @sigma_lb.setter
    def sigma_lb(self, value):
        value=self.__check_array__(value)
        self._sigma_lb=value
        self._theta_lb['sigma']=value
    #--------------------------------theta--------------------------------------#
    @property
    def theta(self):
        return np.concatenate(list(self._theta.values()))
    
    @theta.setter
    def theta(self, value):
        self.sigma=value
    
    @property
    def theta_ub(self):
        return np.concatenate(list(self._theta_ub.values()))
    
    @theta_ub.setter
    def theta_ub(self, value):
        self.sigma_ub=value
    
    @property
    def theta_lb(self):
        return np.concatenate(list(self._theta_lb.values()))
    
    @theta_lb.setter
    def theta_lb(self, value):
        self.sigma_lb=value