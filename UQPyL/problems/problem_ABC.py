import abc
import numpy as np
from typing import Union
class ProblemABC(metaclass=abc.ABCMeta):

    def __init__(self, n_input:int, n_output:int, ub: Union[int, float, np.ndarray], lb: Union[int, float, np.ndarray], disc_var=None, cont_var=None, x_labels=None):
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        self.disc_var=disc_var
        self.cont_var=cont_var
        
        if x_labels is None:
            self.x_labels=['x_'+str(i) for i in range(1,n_input+1)]
    
    @abc.abstractmethod
    def evaluate(self,X):
        pass
    
    def _unit_X_transform_to_bound(self, X):
        
        X_min=X.min(axis=0)
        X_max=X.max(axis=0)
        
        X_scaled=(X - X_min) / (X_max - X_min)
        
        return X_scaled*(self.ub-self.lb)+self.lb
    
    
    def _set_ub_lb(self,ub: Union[int, float, np.ndarray], lb: Union[int, float, np.ndarray]) -> None:
        
        if (isinstance(ub,(int, float))):
            self.ub=np.ones((1,self.n_input))*ub
        elif(isinstance(ub,np.ndarray)):
            
            self._check_bound(ub)
            self.ub=ub
            
        if (isinstance(lb,(int, float))):
            self.lb=np.ones((1,self.n_input))*lb
        elif(isinstance(lb,np.ndarray)):
            
            self._check_bound(lb)
            self.lb=lb 
    def _check_2d(self, X:np.ndarray):
        return np.atleast_2d(X)
    
    def _check_bound(self,bound: np.ndarray):
        
        if(not bound.shape[1]==self.n_input):
            raise ValueError('the input bound is inconsistent with the input n_inputensions')
        