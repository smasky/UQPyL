# class krg_kernels():
import numpy as np
from typing import Union

class Krg_Kernel():
    _n_input=None
    def __init__(self, theta: Union[float, np.ndarray]=1, 
                 theta_lb: Union[float, np.ndarray]=0, theta_ub: Union[float, np.ndarray]=1e5,
                 heterogeneous: bool=True
                 ):
        
        self.theta=theta
        self.theta_lb=theta_lb
        self.theta_ub=theta_ub
        self.heterogenous=heterogeneous
        
    def __check_array__(self, value):
        
        if isinstance(value, (float, int)):
            tmp=np.ones(self.n_input)
            value_=np.full(tmp, value)
        elif isinstance(value, np.ndarray):
            if value.ndim>1:
                value_=value.ravel()
                if value_.shape[0]!=self.n_input:
                    raise ValueError("Please make sure the shape of value")
        else:
            raise ValueError("Please make sure the type of value")

        return value_
    
    ##################################Attribute############################################
    #--------------------------------length_scale----------------------------------------#
    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, value):
        value=self.__check_array__(value)
        self._theta=value
    
    @property
    def theta_lb(self):
        return self._theta_lb
    
    @theta_lb.setter
    def theta_lb(self, value):
        value=self.__check_array__(value)
        self._theta_lb=value
    
    @property
    def theta_ub(self):
        return self._theta_ub
    
    @theta_ub.setter
    def theta_ub(self, value):
        value=self.__check_array__(value)
        self._theta_ub=value
    
    @property
    def n_input(self):
        return self._n_input
    
    @n_input.setter
    def n_input(self, value):
        if not isinstance(value, int):
            raise ValueError("Please the type of value is int")
        
        if self.heterogenous:
            if self.theta.shape[0]==1:
                self.theta = np.resize(self.theta, (value, 1)) 
                self.theta_lb = np.resize(self.theta_lb, (value, 1))
                self.theta_ub = np.resize(self.theta_ub, (value, 1))
        self._n_input=value