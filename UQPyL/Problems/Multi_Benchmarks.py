from typing import Any
from .Problem_ABC import ProblemABC
import numpy as np
from typing import Union


class ZDT1(ProblemABC):
    
    def __init__(self, dim:int =30, NOutput: int=2, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None):
        self.dim=dim
        self.NOutput=NOutput
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
    
    def evaluate(self, X, unit=False):
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        Y=np.zeros((X.shape[0], self.NOutput))
        Y[:,0]=X[:,0]
        g=1+9*np.mean(X[:, 1:], axis=1)
        h=1-np.sqrt(Y[:,0]/g)
        Y[:,1]=g*h
        return Y
    
    def get_optimum(self, N):
        R=np.zeros((N,self.NOutput))
        R[:,0]=np.linspace(0,1,N)
        R[:,1]=1-np.sqrt(R[:,0])
        return R