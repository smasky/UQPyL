import numpy as np
from typing import Union, Optional

from .base_kernel import Krg_Kernel

class Guass_Kernel(Krg_Kernel):
    
    def __init__(self, theta: Union[float, np.ndarray]=1, 
                 theta_lb: Union[float, np.ndarray]=0, theta_ub: Union[float, np.ndarray]=1e5):
        
        super().__init__(theta, theta_lb, theta_ub)
    
    def __call__(self, D: np.ndarray):
        '''
            Parameters:
                D: np.ndarray
                    The distance matrix
        '''
        
        td = D * -self.theta
        r = np.exp(np.sum(D * td, axis=1))
    
        return r
    
    ##################################Attribute############################################
    #--------------------------------length_scale----------------------------------------#
    # @property
    # def theta(self):
    #     return self._theta
    
    # @theta.setter
    # def theta(self, value):
    #     value=self.__check_array__(value)
    #     self._theta=value
    
    # @property
    # def theta_lb(self):
    #     return self._theta_lb
    
    # @theta.setter
    # def theta_lb(self, value):
    #     value=self.__check_array__(value)
    #     self._theta_lb=value
    
    # @property
    # def theta_ub(self):
    #     return self._theta_ub
    
    # @theta.setter
    # def theta_ub(self, value):
    #     value=self.__check_array__(value)
    #     self._theta_ub=value
    
    # @property
    # def n_input(self):
    #     return self._n_input
    
    # @theta.setter
    # def n_input(self, value):
        
    #     self._n_input=value
    
        
        