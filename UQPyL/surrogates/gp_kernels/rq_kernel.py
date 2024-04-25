from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np

from .base_kernel import Gp_Kernel


class RationalQuadratic(Gp_Kernel):
    """
    Constant Kernel
    
    Attribute:
    
    theta: the set of unknown parameters. np.vstack(length_scale, alpha)

    """
    def __init__(self, length_scale: float=1.0, l_ub: float=10.0, l_lb:float=0.0, 
                 alpha: float=1.0, alpha_ub: float=1e5, alpha_lb: float=1e-5):
        
        super().__init__()
        if not isinstance(l_ub, type(l_lb)) and not isinstance(length_scale, type(l_ub)):
            raise ValueError("the type of length_scale, ub and lb is not consistent")
        
        # if not isinstance(l_ub, float):
        #     raise ValueError("the RationalQuadratic Kernel only support scale length_scale")
        
        self.length_scale=length_scale
        self.l_ub=l_ub; self.l_lb=l_lb
        self.alpha=alpha
        self.alpha_ub=alpha_ub;self.alpha_lb=alpha_lb
   
    def __call__(self, trainX: np.ndarray, trainY: Optional[np.ndarray]=None):
        
        if trainY is None:
            dists=squareform(pdist(trainX, metric="sqeuclidean"))
            tmp= dists / (2*self.alpha* self.length_scale**2)
            base=1 + tmp
            K=base**-self.alpha
            np.fill_diagonal(K,1)
        else:
            dists=cdist(trainX, trainY, metric="sqeuclidean")
            K= (1+dists / (2* self.alpha * self.length_scale**2)) ** -self.alpha
        
        return K
    #################################Attributes##########3########################
    #------------------------------length_scale-----------------------------------#
    @property
    def length_scale(self):
        return self._length_scale
    
    @length_scale.setter
    def length_scale(self, value):
        value=self.__check_array__(value)
        self._length_scale=value.copy()
        self._theta['length_scale']=value.copy()
        
    @property
    def l_ub(self):
        return self._l_ub
    
    @l_ub.setter
    def l_ub(self, value):
        value=self.__check_array__(value)
        self._l_ub=value
        self._theta_ub['length_scale']=value
        
    @property
    def l_lb(self):
        return self._l_lb
    
    @l_lb.setter
    def l_lb(self, value):
        value=self.__check_array__(value)
        self._l_lb=value
        self._theta_lb['length_scale']=value
    #---------------------------------alpha-----------------------------------#
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        value=self.__check_array__(value)
        self._alpha=value
        self._theta['alpha']=value

    @property
    def alpha_ub(self):
        return self._alpha_ub
    
    @alpha_ub.setter
    def alpha_ub(self, value):
        value=self.__check_array__(value)
        self._alpha_ub=value
        self._theta_ub['alpha']=value
    
    @property
    def alpha_lb(self):
        return self._alpha_lb
    
    @alpha_lb.setter
    def alpha_lb(self, value):
        value=self.__check_array__(value)
        self._theta_lb['alpha']=value
        self._alpha_lb=value
    #--------------------------------theta----------------------------------#
    @property
    def theta(self):
        return np.concatenate(list(self._theta.values()))
    
    @theta.setter
    def theta(self, value):
        self.alpha=value[-1]
        self.length_scale=value[:-1]
    
    @property
    def theta_ub(self):
        return np.concatenate(list(self._theta_ub.values()))
    
    @theta_ub.setter
    def theta_ub(self, value):
        self.alpha_ub=value[-1]
        self.l_ub=value[:-1]
    
    @property
    def theta_lb(self):
        return np.concatenate(list(self._theta_lb.values()))
    
    @theta_lb.setter
    def theta_lb(self, value):
        self.alpha_lb=value[-1]
        self.l_lb=value[:-1]