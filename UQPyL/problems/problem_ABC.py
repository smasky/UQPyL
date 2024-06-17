import abc
import numpy as np
from typing import Union
class ProblemABC(metaclass=abc.ABCMeta):

    def __init__(self, n_input:int, n_output:int, ub: Union[int, float, np.ndarray], lb: Union[int, float, np.ndarray], disc_var=None, disc_range=None, cont_var=None, x_labels=None):
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        if disc_var is None:
            self.disc_var=[0]*n_input
            self.disc_range=[0]*n_input
        else:
            self.disc_var=disc_var
            self.disc_range=disc_range
        self.cont_var=cont_var
        
        if x_labels is None:
            self.x_labels=['x_'+str(i) for i in range(1,n_input+1)]
    
    @abc.abstractmethod
    def evaluate(self,X):
        pass
    
    def _discrete_variable_transform(self, X):
        if self.disc_var is not None:
            for i in range(self.n_input):
                if self.disc_var[i]==1:
                    num_interval=len(self.disc_range[i])
                    bins=np.linspace(self.lb[i], self.ub[i], num_interval+1)
                    indices = np.digitize(X[:, i], bins[1:])
                    
                    if isinstance(self.disc_range[i], list):
                        X[:, i]=np.array(self.disc_range[i])[indices]
                    else:
                        X[:, i]=self.disc_range[i][indices]
    
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
        