import numpy as np
import abc
from typing import Any, Optional, Union
from scipy.spatial.distance import pdist, squareform, cdist
class Kernel():
    def __init__(self):
        self._theta={}
        self._theta_ub={}
        self._theta_lb={}
    
    def __check_array__(self, value: Union[float,np.ndarray]):
        
        if isinstance(value, float):
            value=np.array([value])
        elif isinstance(value, np.ndarray):
            if value.ndim>1:
                value=value.ravel()
        else:
            raise ValueError("Please make sure the type of value")
        
        return value
        
class RBF(Kernel):
    """
        RBF kernel
    """
    def __init__(self, length_scale: Union[float, np.ndarray]=1.0,
                 l_ub: Union[float, np.ndarray]=1e5,l_lb: Union[float, np.ndarray]=1e-5):
        
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
        
class Matern(Kernel):
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
        
class RationalQuadratic(Kernel):
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
        
class Constant(Kernel):
    """
    Constant
    """
    def __init__(self, c: float=1.0):
        self.c=c
    
    def setHyperPara(self, theta: np.ndarray):
        self.c=theta[0]
    
    def __call__(self, trainX: np.ndarray, trainY: Optional[np.ndarray]=None):
    
        if trainY is None:
            K=np.ones((trainX.shape[0],trainY.shape[0]))*self.c
        else:
            K=np.ones((trainX.shape[0], trainX.shape[0]))*self.c
        
        return K

class DotProduct(Kernel):
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
    
    
    
    
     
    


         
            
        
        
        
    