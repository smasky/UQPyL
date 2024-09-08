import numpy as np
from typing import Union

from ..problem_ABC import ProblemABC
##-----------Reference------------------##
# E. Zitzler, K. Deb, and L. Thiele, Comparison of multiobjective
# evolutionary algorithms: Empirical results, Evolutionary computation,
# 2000, 8(2): 173-195.
#--------------------------------------##
class ZDT1(ProblemABC):
    
    def __init__(self, n_input:int =30, n_output: int=2, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None):
        
        super().__init__(n_input, n_output, ub, lb, disc_var, cont_var)
        
        if n_output!=2:
            raise ValueError("ZDT1 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        Y=np.zeros((X.shape[0], self.n_output))
        Y[:,0]=X[:,0]
        g=1+9*np.mean(X[:, 1:], axis=1)
        h=1-np.sqrt(Y[:,0]/g)
        Y[:,1]=g*h
        
        return Y
    
    def get_optimum(self, N):
        
        R=np.zeros((N,self.n_output))
        R[:,0]=np.linspace(0,1,N)
        R[:,1]=1-np.sqrt(R[:,0])
        
        return R
    
    def get_PF(self):
        
        R=self.get_optimum(100)
        
        return R

class ZDT2(ProblemABC):
    
    def __init__(self, n_input:int =30, n_output: int=2, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None):
        
        super().__init__(n_input, n_output, ub, lb, disc_var, cont_var)
        
        if n_output!=2:
            raise ValueError("ZDT2 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        Y=np.zeros((X.shape[0], self.n_output))
        Y[:,0]=X[:,0]
        g=1+9*np.sum(X[:, 1:], axis=1)/(self.n_input-1)
        h=1-(Y[:,0]/g)**2
        Y[:,1]=g*h
        
        return Y
    
    def get_optimum(self, N):
        
        R=np.zeros((N,self.n_output))
        R[:,0]=np.linspace(0,1,N)
        R[:,1]=1-(R[:,0])**2
        
        return R
    
    def get_PF(self):
        
        R=self.get_optimum(100)
        
        return R
    
class ZDT3(ProblemABC):
    
    def __init__(self, n_input:int =30, n_output: int=2, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None):
        
        super().__init__(n_input, n_output, ub, lb, disc_var, cont_var)
        
        if n_output!=2:
            raise ValueError("ZDT4 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        Y=np.zeros((X.shape[0], self.n_output))
        Y[:,0]=X[:,0]
        g=1+9*np.sum(X[:, 1:], axis=1)/(self.n_input-1)
        h=1-np.sqrt(Y[:,0]/g)-(Y[:,0]/g)*np.sin(10*np.pi*Y[:,0])
        Y[:,1]=g*h
        
        return Y
    
    def get_optimum(self, N):
        
        from ..utility_functions._NDsort import NDSort
        
        R=np.zeros((N, self.n_output))
        R[:,0]=np.linspace(0,1,N)
        R[:,1]=1 - np.sqrt(R[:,0]) - R[:, 0] * np.sin(10 * np.pi * R[:, 0])
        
        FrontNo,_=NDSort(R,1)
        R[FrontNo>1,:] = np.nan
        
        return R
    
    def get_PF(self):
        
        R=self.get_optimum(300)
        
        return R

class ZDT4(ProblemABC):
    
    def __init__(self, n_input:int =30, n_output: int=2, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None):
       
        super().__init__(n_input, n_output, ub, lb, disc_var, cont_var)
        
        if n_output!=2:
            raise ValueError("ZDT4 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
            
        Y = np.zeros((X.shape[0], 2))
        Y[:, 0] = X[:, 0]
        g = 1 + 10 * (X.shape[1] - 1) + np.sum(X[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * X[:, 1:]), axis=1)
        h = 1 - (X[:, 0] / g) ** 0.5
        Y[:, 1] = g * h
        
        return Y
    
    def get_optimum(self, N):
        
        R = np.zeros((N, 2))
        R[:, 0] = np.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0] ** 0.5
        
        return R

    def GetPF(self):
        
        R = self.get_optimum(100)
        
        return R

class ZDT6(ProblemABC):
    
    def __init__(self, n_input:int =30, n_output: int=2, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None):
        
        super().__init__(n_input, n_output, ub, lb, disc_var, cont_var)
        
        if n_output!=2:
            raise ValueError("ZDT6 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        Y=np.zeros((X.shape[0], self.n_output))
        Y[:,0]=1-np.exp(-4*X[:,0])*np.sin(6*np.pi*X[:,0])**6
        g=1+9*np.sum(X[:, 1:], axis=1)/(self.n_input-1)**0.25
        h=1-(Y[:,0]/g)**2
        Y[:,1]=g*h
        
        return Y
    
    def get_optimum(self, N):
        
        min=0.280775
        R=np.zeros((N,self.n_output))
        R[:,0]=np.linspace(min,1,N)
        R[:,1]=1-R[:,0]**2
        
        return R
    
    def get_PF(self):
        
        R=self.get_optimum(100)
        
        return R