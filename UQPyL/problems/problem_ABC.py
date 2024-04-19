import abc
import numpy as np
from typing import Union
class ProblemABC(metaclass=abc.ABCMeta):
    n_input=None
    ub=None
    lb=None
    disc_var=None
    cont_var=None
    x_labels=None
    @abc.abstractmethod
    def evaluate(self,X):
        pass
    
    def _unit_X_transform_to_bound(self,X):
        
        return X*(self.ub-self.lb)+self.lb
    
    
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
        