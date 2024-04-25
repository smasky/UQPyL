from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np

from .base_kernel import Gp_Kernel

class RBF(Gp_Kernel):
    """
        RBF kernel
    """
    def __init__(self, length_scale: Union[float, np.ndarray]=1.0,
                 l_ub: Union[float, np.ndarray]=1e5, l_lb: Union[float, np.ndarray]=1e-5):
        
        super().__init__()
        
        if not isinstance(l_ub, type(l_lb)) and not isinstance(length_scale, type(l_ub)):
            raise ValueError("the type of length_scale, ub and lb is not consistent")
        
        if isinstance(length_scale, float):
            length_scale=np.array([length_scale])
            l_lb=np.array([l_lb])
            l_ub=np.array([l_ub])
        
        self.length_scale=length_scale
        self.l_lb=l_lb
        self.l_ub=l_ub

    def __call__(self, trainX1: np.ndarray, trainX2: Optional[np.ndarray]=None):
        
        if trainX2 is None:
            dists=pdist(trainX1/self.length_scale, metric="sqeuclidean")
            K=squareform(np.exp(-0.5*dists))
            np.fill_diagonal(K,1.0)
        else:
            dists=cdist(trainX1/self.length_scale, trainX2/self.length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
        return K
    
    ##################################Attribute############################################
    #--------------------------------length_scale----------------------------------------#
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
    #--------------------------------------theta-------------------------------------------#
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