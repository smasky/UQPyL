import abc
import numpy as np
from typing import Union
class ProblemABC(metaclass=abc.ABCMeta):

    def __init__(self, n_input:int, n_output:int, ub: Union[int, float, np.ndarray], lb: Union[int, float, np.ndarray], var_type=None, var_set=None, x_labels=None, y_labels=None):
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        if var_type is None:
            self.var_type=np.array([0]*n_input)
        else:
            self.var_type=np.array(var_type)
        
        if var_set is None:
            self.var_set={}
        else:
            self.var_set=var_set
        
        if x_labels is None:
            self.x_labels=['x_'+str(i) for i in range(1,n_input+1)]

        if y_labels is None:
            self.y_labels=['y_'+str(i) for i in range(1,n_output+1)]
        
    @abc.abstractmethod
    def evaluate(self,X):
        pass
    
    def _transform_special_parameters(self, X):
        
        int_indice=np.where(self.var_type==1)[0]
        if int_indice.size>0:
            X[:, int_indice]=np.round(X[:, int_indice])
        
        if self.var_set is not None:
            for index, set in self.var_set.items():
                num_interval=len(set)
                bins=np.linspace(self.lb[index], self.ub[index], num_interval+1)
                indices = np.digitize(X[:, index], bins) - 1
                X[:, index]=np.array([set[i] for i in indices])
                    
        return X
    
    def _unit_X_transform_to_bound(self, X):
        
        X_min=X.min(axis=0)
        X_max=X.max(axis=0)
        
        X_scaled=(X - X_min) / (X_max - X_min)
        X_scaled=X_scaled*(self.ub-self.lb)+self.lb
        X_scaled=self._transform_special_parameters(X_scaled)
        return X_scaled
    
    
    def _set_ub_lb(self,ub: Union[int, float, np.ndarray], lb: Union[int, float, np.ndarray]) -> None:
        
        if (isinstance(ub,(int, float))):
            self.ub=np.ones((1,self.n_input))*ub
        elif(isinstance(ub,np.ndarray)):
            self._check_bound(ub)
            self.ub=ub.reshape(1, -1)
            
        if (isinstance(lb,(int, float))):
            self.lb=np.ones((1,self.n_input))*lb
        elif(isinstance(lb,np.ndarray)):
            self._check_bound(lb)
            self.lb=lb.reshape(1, -1)
            
    def _check_2d(self, X:np.ndarray):
        return np.atleast_2d(X)
    
    def _check_bound(self,bound: np.ndarray):
        
        bound=bound.ravel()
        if(not bound.shape[0]==self.n_input):
            raise ValueError('the input bound is inconsistent with the input n_inputensions')
        