import numpy as np
from scipy.spatial.distance import pdist,squareform
import abc

class BaseKernel(metaclass=abc.ABCMeta):
    
    n_samples=None
    n_features=None
    
    def __init__(self):
        
        self._theta={}
        self._theta_ub={}
        self._theta_lb={}
        self.setting=Setting(self.name)
        
    def __check_array__(self, value):
            
        if isinstance(value, float):
            value=np.array([value])
        elif isinstance(value, np.ndarray):
            if value.ndim>1:
                value=value.ravel()
        else:
            raise ValueError("Please make sure the used type (float or np.ndarray) of value")
        
        return value
    
    
    def evaluate(self,pdist):
        pass
    
    def get_A_Matrix(self,train_X):
        dist=squareform(pdist(train_X,'euclidean'))
        Phi=self.evaluate(dist)
        
        self.n_samples, self.n_features=train_X.shape
        
        S,Tail=self.get_Tail_Matrix(train_X)
        if(S):
            temp1=np.hstack((Phi,Tail))
            t1=Tail.transpose()
            t2=np.zeros((self.get_degree(self.n_features),self.get_degree(self.n_features)))
            
            temp2=np.hstack((Tail.transpose(),np.zeros((self.get_degree(self.n_features),self.get_degree(self.n_features)))))
            A_Matrix=np.vstack((temp1,temp2))
        else:
            A_Matrix=Phi
        return A_Matrix
            
    def get_Tail_Matrix(self,train_X): 
        return (False,None)
    
    def get_degree(self,n_samples):
        return None
    
    def setParameters(self, key, value, lb, ub):
        
        self.setting.setParameter(key, value, lb, ub)
    
    def getParaValue(self, *args):
        
        return self.setting.getParaValue(*args) 
class Setting():
    
    def __init__(self, prefix):
        
        self.prefix=prefix
        self.hyperParas={}
        self.paraUb={}
        self.paraLb={}
    
    def keys(self):
        
        return self.hyperParas.keys()
    
    def values(self):
        
        return self.hyperParas.values()
    
    def setParameter(self, key, value, lb, ub):
        
        self.hyperParas[key]=value
        self.paraLb[key]=lb
        self.paraUb[key]=ub

    def getParaValue(self, *args):
        
        values=[]
        for arg in args:
            values.append(self.hyperParas[arg])
        
        if len(args)>1:
            return tuple(values)
        else:
            return values[0]
    
# class Cubic(Kernel):
#     """
#         Cubic Kernel
#     """
#     name="Cubic"
#     def __init__(self, epsilon: float=1.0):
        
#         super().__init__()
    
#     def evaluate(self, dist: np.ndarray):
        
#         return np.power(dist, 3)
    
#     def get_Tail_Matrix(self, train_X: np.ndarray):
#         Tail=np.ones((self.n_samples, self.n_features+1))
#         Tail[:self.n_samples, :self.n_features]=train_X.copy()
#         return (True,Tail)
    
#     def get_degree(self, n_samples: int):
#         return n_samples+1
#     ##############################Attribute##########################
#     @property
#     def theta(self):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @theta.setter
#     def theta(self, value):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @property
#     def theta_ub(self):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @theta_ub.setter
#     def theta_ub(self, value):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @property
#     def theta_lb(self):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @theta_lb.setter
#     def theta_lb(self, value):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
# class Gaussian(Kernel):
#     """
#         Gaussian Kernel
        
#         theta: gamma
#     """
    
#     name="Gaussian"
    
#     def __init__(self, gamma: float=1.0, gamma_ub: float=1e5, gamma_lb: float=1e-5):
        
#         super().__init__()
        
#         self.gamma=gamma
#         self.gamma_ub=gamma_ub
#         self.gamma_lb=gamma_lb
        
#     def evaluate(self,dist):
        
#         return np.exp(-1*self.gamma*np.power(dist,2))
    
#     ###########################Attribute###################################
#     #----------------------------gamma------------------------------------#
#     @property
#     def gamma(self):
        
#         return self._gamma
    
#     @gamma.setter
#     def gamma(self, value):
        
#         value=self.__check_array__(value)
#         self._gamma=value
#         self._theta['gamma']=value
    
#     @property
#     def gamma_ub(self):
        
#         return self._gamma_ub
    
#     @gamma_ub.setter
#     def gamma_ub(self, value):
        
#         value=self.__check_array__(value)
#         self._gamma_ub=value
#         self._theta_ub['gamma']=value
        
#     @property
#     def gamma_lb(self):
        
#         return self._gamma_lb
    
#     @gamma_lb.setter
#     def gamma_lb(self, value):
        
#         value=self.__check_array__(value)
#         self._gamma_lb=value
#         self._theta_lb['gamma']=value
#     #------------------------------theta--------------------------#
#     @property
#     def theta(self):
#         return np.concatenate(list(self._theta.values()))
    
#     @theta.setter
#     def theta(self, value):
#         self.gamma=value
    
#     @property
#     def theta_ub(self):
#         return np.concatenate(list(self._theta_ub.values()))
    
#     @theta_ub.setter
#     def theta_ub(self, value):
#         self.gamma_ub=value
    
#     @property
#     def theta_lb(self):
#         return np.concatenate(list(self._theta_lb.values())) 
    
#     @theta_lb.setter
#     def theta_lb(self, value):
#         self.gamma_lb=value
    

# class Gaussian(Kernel):
    
#     name="Gaussian"
    
#     def __init__(self, gamma: float=1.0, gamma_ub: float=1e5, gamma_lb: float=1e-5):
        
#         super().__init__()
        
#         self.gamma=gamma
#         self.gamma_ub=gamma_ub
#         self.gamma_lb=gamma_lb
        
#     def evaluate(self,dist):
        
#         return np.exp(-1*self.gamma*np.power(dist,2))
    
#     ###########################Attribute###################################
#     #----------------------------gamma------------------------------------#
#     @property
#     def gamma(self):
        
#         return self._gamma
    
#     @gamma.setter
#     def gamma(self, value):
        
#         value=self.__check_array__(value)
#         self._gamma=value
#         self._theta['gamma']=value
    
#     @property
#     def gamma_ub(self):
        
#         return self._gamma_ub
    
#     @gamma_ub.setter
#     def gamma_ub(self, value):
        
#         value=self.__check_array__(value)
#         self._gamma_ub=value
#         self._theta_ub['gamma']=value
        
#     @property
#     def gamma_lb(self):
        
#         return self._gamma_lb
    
#     @gamma_lb.setter
#     def gamma_lb(self, value):
        
#         value=self.__check_array__(value)
#         self._gamma_lb=value
#         self._theta_lb['gamma']=value
#     #------------------------------theta--------------------------#
#     @property
#     def theta(self):
#         return np.concatenate(list(self._theta.values()))
    
#     @theta.setter
#     def theta(self, value):
#         self.gamma=value
    
#     @property
#     def theta_ub(self):
#         return np.concatenate(list(self._theta_ub.values()))
    
#     @theta_ub.setter
#     def theta_ub(self, value):
#         self.gamma_ub=value
    
#     @property
#     def theta_lb(self):
#         return np.concatenate(list(self._theta_lb.values())) 
    
#     @theta_lb.setter
#     def theta_lb(self, value):
#         self.gamma_lb=value

# class Linear(Kernel):
#     """
#     Linear Kernel
#     """
#     name="Linear"
#     def evaluate(self,dist):
        
#         return -dist
#     #TODO check the formula
#     def get_degree(self,n_samples):
        
#         return 1
    
#     def get_Tail_Matrix(self,train_X):
        
#         return (True,np.ones((train_X.shape[0],1)))
    
#     ###########################Attribute###################################
#     @property
#     def theta(self):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @theta.setter
#     def theta(self, value):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @property
#     def theta_ub(self):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @theta_ub.setter
#     def theta_ub(self, value):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @property
#     def theta_lb(self):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @theta_lb.setter
#     def theta_lb(self, value):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")

# class Multiquadric(Kernel):
#     """
#     Multiquaric kernel
    
#     """
#     name="Multiquadric"
#     def __init__(self, gamma: float=1.0, gamma_ub: float=1e5, gamma_lb: float=1e-5):
        
#         super().__init__()
        
#         self.gamma=gamma
#         self.gamma_ub=gamma_ub
#         self.gamma_lb=gamma_lb
        
#     def evaluate(self,dist):
        
#         return np.sqrt(np.power(dist,2)+self.gamma*self.gamma)

#     def get_degree(self,n_samples):
        
#         return 1
    
#     def get_Tail_Matrix(self,train_X):
        
#         return (True,np.ones((train_X.shape[0],1)))
#     ###########################Attribute###################################
#     #----------------------------gamma------------------------------------#
#     @property
#     def gamma(self):
        
#         return self._gamma
    
#     @gamma.setter
#     def gamma(self, value):
        
#         value=self.__check_array__(value)
#         self._gamma=value
#         self._theta['gamma']=value
    
#     @property
#     def gamma_ub(self):
        
#         return self._gamma_ub
    
#     @gamma_ub.setter
#     def gamma_ub(self, value):
        
#         value=self.__check_array__(value)
#         self._gamma_ub=value
#         self._theta_ub['gamma']=value
        
#     @property
#     def gamma_lb(self):
        
#         return self._gamma_lb
    
#     @gamma_lb.setter
#     def gamma_lb(self, value):
        
#         value=self.__check_array__(value)
#         self._gamma_lb=value
#         self._theta_lb['gamma']=value
#     #----------------------------------theta----------------------------#
#     @property
#     def theta(self):
#         return np.concatenate(list(self._theta.values()))
    
#     @theta.setter
#     def theta(self, value):
#         self.gamma=value
    
#     @property
#     def theta_ub(self):
#         return np.concatenate(list(self._theta_ub.values()))
    
#     @theta_ub.setter
#     def theta_ub(self, value):
#         self.gamma_ub=value
    
#     @property
#     def theta_lb(self):
#         return np.concatenate(list(self._theta_lb.values())) 
    
#     @theta_lb.setter
#     def theta_lb(self, value):
#         self.gamma_lb=value

# class Thin_plate_spline(Kernel):
#     """
    
#     Thin_plate_spline
    
#     """
#     name="Thin_plate_spline"
#     def __init__(self):
#         super().__init__()
    
#     def evaluate(self,dist):
        
#         dist[dist < np.finfo(float).eps] = np.finfo(float).eps
#         return np.power(dist,2)*np.log(dist)
    
#     def get_Tail_Matrix(self,train_X):
        
#         Tail=np.ones((self.n_samples,self.n_features+1))
#         Tail[:self.n_samples,:self.n_features]=train_X.copy()
#         return (True,Tail)
    
#     def get_degree(self,n_samples):
        
#         return n_samples+1
    
#     ###########################Attribute###################################
#     @property
#     def theta(self):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @theta.setter
#     def theta(self, value):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @property
#     def theta_ub(self):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @theta_ub.setter
#     def theta_ub(self, value):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @property
#     def theta_lb(self):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
#     @theta_lb.setter
#     def theta_lb(self, value):
#         raise ValueError("There is no hyper-parameters in Cubic kernel")
    
    
    
    
    
            
    
    
