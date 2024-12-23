import abc
import numpy as np
from typing import Union
class ProblemABC(metaclass=abc.ABCMeta):

    def __init__(self, nInput:int, nOutput:int, 
                 ub: Union[int, float, np.ndarray], lb: Union[int, float, np.ndarray], 
                 var_type=None, var_set=None, 
                 x_labels=None, y_labels=None):
        
        self.nInput=nInput
        self.nOutput=nOutput
        self._set_ub_lb(ub,lb)
        self.encoding="real"
        
        if var_type==None or var_type=={}:
            self.var_type=np.zeros(self.nInput)
            self.I_float=np.arange(self.nInput)
            
        else:
            self.encoding="mix"
            self.var_type=np.array(var_type, dtype=np.int32)
            self.I_float=np.where(self.var_type==0)[0]
            self.I_int=np.where(self.var_type==1)[0]
            self.I_dst=np.where(self.var_type==2)[0]

        if var_set is None:
            self.var_set={}
        else:
            self.var_set={}
            
            for i in self.I_dst:
                if isinstance(var_set[i], list):
                    self.var_set[i]=var_set[i]
                else:
                    raise ValueError("The type of sub var_set must be list.")
        
        if x_labels is None:
            self.x_labels=['x_'+str(i) for i in range(1,nInput+1)]
        else:
            self.x_labels=x_labels

        if y_labels is None:
            self.y_labels=['y_'+str(i) for i in range(1,nOutput+1)]
        else:
            self.y_labels
        
    @abc.abstractmethod
    def evaluate(self, X):
        pass
    
    def constraint(self, X):
        
        return np.zeros((X.shape[0], 1))
    
    def getOptimum(self):
        pass
    
    def _transform_discrete_var(self, X):
        
        for i in self.I_dst:
            S=self.var_set[i]
            num_interval=len(S)
            bins=np.linspace(self.lb[0, i], self.ub[0, i], num_interval+1)
            indices = np.digitize(X[:, i], bins, right=False) - 1
            indices[indices==num_interval]=num_interval-1
            X[:, i]=np.array([S[i] for i in indices])
            
        return X
    
    def _transform_int_var(self, X):
        
        X[:, self.I_int]=np.round(X[:, self.I_int])

        return X
    
    def _unit_X_transform_to_bound(self, X, dst=True):
        
        X_min=X.min(axis=0)
        X_max=X.max(axis=0)
        
        X_scaled=(X - X_min) / (X_max - X_min)
        X_scaled=X_scaled*(self.ub-self.lb)+self.lb
        
        if self.encoding=='mix':
            
            self._transform_int_var(X_scaled)
                
            if dst:
                X_scaled=self._transform_discrete_var(X_scaled)
        
        return X_scaled 
    
    def _set_ub_lb(self,ub: Union[int, float, np.ndarray], lb: Union[int, float, np.ndarray]) -> None:
        
        if (isinstance(ub, (int, float))):
            self.ub=np.ones((1,self.nInput))*ub
            
        elif(isinstance(ub, np.ndarray)):
            self._check_bound(ub)
            self.ub=ub.reshape(1, -1)
        
        elif(isinstance(ub, list)):
            self.ub=np.array(ub)
            self._check_bound(self.ub)
            
        else:
            raise ValueError("The type of ub is not supported.")
        
        if (isinstance(lb, (int, float))):
            self.lb=np.ones((1,self.nInput))*lb
            
        elif(isinstance(lb, np.ndarray)):
            self._check_bound(lb)
            self.lb=lb.reshape(1, -1)
        
        elif(isinstance(lb, list)):
            self.lb=np.array(lb)
            self._check_bound(self.lb)
        
        else:
            raise ValueError("The type of lb is not supported.")
        
    def _check_2d(self, X:np.ndarray):
        return np.atleast_2d(X)
    
    def _check_bound(self,bound: np.ndarray):
        
        bound=bound.ravel()
        if(not bound.shape[0]==self.nInput):
            raise ValueError('the input bound is inconsistent with the input nInputensions')