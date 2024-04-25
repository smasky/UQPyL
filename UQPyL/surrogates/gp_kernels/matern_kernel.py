from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np

from .base_kernel import Gp_Kernel


class Matern(Gp_Kernel):
    """
       Matern Kernel
       
       Parameters:
       
       nu: this parameter determine the smooth of the prediction
       
       length_scale:  a scaler or vector to determine the correlation of the input data
       
       ub, lb: the upper or lower bound of the length_scale
       
       Attribute:
       
       theta: the set of unknown parameters 
 
    """
    def __init__(self, length_scale: Union[float, np.ndarray]=1.0,
                 l_ub: Union[float, np.ndarray]=1e5, l_lb: Union[float, np.ndarray]=1e-5,
                 nu: float=1.5):
                
        super().__init__()
        
        if not isinstance(l_ub, type(l_lb)) and not isinstance(length_scale, type(l_ub)):
            raise ValueError("the type of length_scale, ub and lb is not consistent")
        
        self.length_scale=length_scale
        self.l_ub=l_ub
        self.l_lb=l_lb
        self.nu=nu
    
    def __call__(self, trainX1: np.ndarray, trainX2: Optional[np.ndarray]=None):
        
        if trainX2 is None:
            dists=pdist(trainX1/self.length_scale, metric="euclidean")
        else:
            dists = cdist(trainX1/self.length_scale, trainX2/self.length_scale, metric="euclidean")
        if self.nu==0.5:
            K=np.exp(-dists)
        elif self.nu==1.5:
            K=dists*np.sqrt(3)
        elif self.nu==2.5:
            K=dists*np.sqrt(5)
            K=(1.0+K+K**2/3.0) * np.exp(-K)
        elif self.nu==np.inf:
            K=np.exp(-(dists**2)/2.0)
        else:
            ##TODO 
            #gamma kv
            raise ValueError("Current do not support other value of nu,\
                             Please use 0.5, 1.5, 2.5, or np.inf.")
        if trainX2 is None:
            K=squareform(K)
            np.fill_diagonal(K,1.0)
        
        return K
    
    ##################################Attribute########################################
    #--------------------------------length_scale------------------------------------#
    @property
    def length_scale(self):
        return self._length_scale
    
    @length_scale.setter
    def length_scale(self, value):
        value=self.__check_array__(value)
        self._length_scale=value
        self._theta['length_scale']=value
    
    @property
    def l_lb(self):
        return self._l_lb
    
    @l_lb.setter
    def l_lb(self, value):
        value=self.__check_array__(value)
        self._l_lb=value
        self._theta_lb['length_scale']=value
        
    @property
    def l_ub(self):
        return self._l_ub
    
    @l_ub.setter
    def l_ub(self, value):
        value=self.__check_array__(value)
        self._l_ub=value
        self._theta_ub['length_scale']=value
    #------------------------------------theta--------------------------------------#
    @property
    def theta(self):
        return np.concatenate(list(self._theta.values()))
    
    @theta.setter
    def theta(self, value):
        self.length_scale=value
    
    @property
    def theta_ub(self):
        return np.concatenate(list(self._theta_ub.values()))
    
    @theta_ub.setter
    def theta_ub(self, value):
        self.l_ub=value
    
    @property
    def theta_lb(self):
        return np.concatenate(list(self._theta_lb.values())) 
    
    @theta_lb.setter
    def theta_lb(self, value):
        self.l_lb=value