import abc
import numpy as np
class ProblemABC(metaclass=abc.ABCMeta):
    dim=None
    ub=None
    lb=None
    disc_var=None
    cont_var=None
    @abc.abstractmethod
    def evaluate(self,X):
        pass
    
    def _unit_X_transform_to_bound(self,X):
        
        return X*(self.ub-self.lb)+self.lb
    
    
    def _set_ub_lb(self,ub: int or float or np.ndarray,lb: int or float or np.ndarray) -> None:
        
        if (isinstance(ub,(int, float))):
            self.ub=np.ones((1,self.dim))*ub
        elif(isinstance(ub,np.ndarray)):
            
            self._check_bound(ub)
            self.ub=ub
            
        if (isinstance(lb,(int, float))):
            self.lb=np.ones((1,self.dim))*lb
        elif(isinstance(lb,np.ndarray)):
            
            self._check_bound(lb)
            self.lb=lb 
    
    def _check_bound(self,bound: np.ndarray):
        
        if(not bound.shape[1]==self.dim):
            raise ValueError('the input bound is inconsistent with the input dimensions')
        