import numpy as np
import itertools
from typing import Union

from .Problem_ABC import ProblemABC

##----------------Reference-------------------#
# K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, Scalable test problems
# for evolutionary multiobjective optimization, Evolutionary multiobjective
# Optimization. Theoretical Advances and Applications, 2005, 105-145.
##--------------------------------------------#
class DTLZ1(ProblemABC):
    
    def __init__(self, dim:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.dim=dim
        self.NOutput=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ1 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
            
        g = 100 * (self.dim - self.n_output + 1 + \
           np.sum((X[:, self.n_output:] - 0.5) ** 2 - \
                  np.cos(20. * np.pi * (X[:, self.n_output:] - 0.5)), axis=1))
        
        Y = 0.5 * np.tile(1 + g, (1, self.n_output)) \
            * np.fliplr(np.cumprod(np.hstack([np.ones((X.shape[0], 1)), X[:, :self.n_output - 1]]), axis=1)) \
            * np.hstack([np.ones((X.shape[0], 1)), 1 - X[:, self.n_output - 1::-1]])
        
        return Y
    
    def get_PF(self):
        
        a = np.linspace(0, 1, 10).reshape(-1, 1)
        R = [a.dot(a.T)/2, a.dot((1 - a.T))/2, (1 - a).dot(np.ones(a.T.shape))/2]
        Y = np.array(list(itertools.product(R[0], R[1], R[2])))
          
        return Y
    
    def get_optimum(self, N):
        
        from .utility_functions._uniformPoint import uniformPoint
        R = uniformPoint(N,self.n_output)/2
        
        return R

class DTLZ2(ProblemABC):
    
    def __init__(self, dim:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.dim=dim
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ2 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        g = np.sum((X[:, self.n_output:] - 0.5) ** 2, axis=1)
        Y = np.tile(1 + g, (1, self.n_output)) \
            * np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1)), np.cos(X[:, :self.n_output - 1] * np.pi / 2))), axis=1)) \
            * np.hstack((np.ones((g.shape[0], 1)), np.sin(X[:, self.n_output - 1::-1] * np.pi / 2)))
        
        return Y
    
    def get_optimum(self, N):
        
        from .utility_functions._uniformPoint import uniformPoint
        R = uniformPoint(N, self.n_output)
        R = R / np.tile(np.sqrt(np.sum(R ** 2, axis=1)).reshape(-1, 1), (1, self.n_output))
        
        return R
    
    def get_PF(self):
        
        a = np.linspace(0, np.pi / 2, 10).reshape(-1, 1)
        R = [np.sin(a) * np.cos(a.T), np.sin(a) * np.sin(a.T), np.cos(a) * np.ones(a.shape).T]
        Y = np.array(list(itertools.product(R[0], R[1], R[2])))
          
        return Y

class DTLZ3(ProblemABC):
    
    def __init__(self, dim:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.dim=dim
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ3 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        g = 100 * (self.dim - self.n_output + 1 + np.sum((X[:, self.n_output:] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, self.n_output:] - 0.5)), axis=1))
        Y = np.tile(1 + g, (1, self.n_output)) * np.fliplr(np.cumprod(np.hstack([np.ones((X.shape[0], 1)), np.cos(X[:, :self.n_output - 1] * np.pi / 2)]), axis=1)) * np.hstack([np.ones((X.shape[0], 1)), np.sin(X[:, self.n_output - 1::-1] * np.pi / 2)])
        return Y
    
    def get_optimum(self, N):
        from .utility_functions._uniformPoint import uniformPoint
        
        R=uniformPoint(N,self.n_output)
        R = R / np.tile(np.sqrt(np.sum(R ** 2, axis=1)), (1, self.n_output))
        return R
    
    def get_PF(self):
        
        a = np.linspace(0, np.pi / 2, 10)
        R = [np.sin(a) * np.cos(a), np.sin(a) * np.sin(a), np.cos(a) * np.ones(a.shape)]
        Y = np.array(list(itertools.product(R[0], R[1], R[2])))
          
        return Y

class DTLZ4(ProblemABC):
    
    def __init__(self, dim:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.dim=dim
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ4 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        X[:, :self.n_output-1] = np.power(X[:, :self.n_output-1], 100)
        g = np.sum(np.power(X[:, self.n_output-1:] - 0.5, 2), axis=1)
        Y = (1 + g[:, np.newaxis]) \
            * np.fliplr(np.cumprod(np.column_stack([np.ones(g.shape[0]), np.cos(X[:, :self.n_output-1] * np.pi / 2)]), axis=1)) \
            * np.column_stack([np.ones(g.shape[0]), np.sin(X[:, self.M-1::-1] * np.pi / 2)])
        
        return Y
    
    def get_optimum(self, N):
        
        from .utility_functions._uniformPoint import uniformPoint
        R = uniformPoint(N, self.n_output)
        R /= np.sqrt(np.sum(R**2, axis=1))[:, np.newaxis]
        return R

    def get_PF(self):

        a = np.linspace(0, np.pi/2, 10)
        R = [np.sin(a) * np.cos(a), np.sin(a) * np.sin(a), np.cos(a) * np.ones_like(a)]
        Y = np.array(list(itertools.product(R[0], R[1], R[2])))
          
        return Y
    
class DTLZ5(ProblemABC):
    
    def __init__(self, dim:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.dim=dim
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ5 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))

        g = np.sum((X[:, self.n_output-1:] - 0.5)**2, axis=1)
        temp = np.repeat(g[:, np.newaxis], self.n_output-2, axis=1)
        X[:, 1:self.n_output-1] = (1 + 2 * temp * X[:, 1:self.n_output-1]) / (2 + 2 * temp)
        Y = (1 + g[:, np.newaxis]) \
                    * np.fliplr(np.cumprod(np.column_stack([np.ones(g.shape[0]), np.cos(X[:, :self.n_output-1] * np.pi / 2)]), axis=1)) \
                    * np.column_stack([np.ones(g.shape[0]), np.sin(X[:, self.n_output-1::-1] * np.pi / 2)])
        return Y
    
    def get_optimum(self, N):
         
        R = np.column_stack([np.linspace(0, 1, N), np.linspace(1, 0, N)])
        R /= np.sqrt(np.sum(R**2, axis=1))[:, np.newaxis]
        R = np.column_stack([np.repeat(R, self.n_output-2, axis=1), R])
        R /= np.sqrt(2) ** np.repeat(np.arange(self.n_output-2, -1, -1), 2, axis=0)
        return R

    def get_pf(self):
        
        return self.get_optimum(100)

class DTLZ6(ProblemABC):
    
    def __init__(self, dim:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.dim=dim
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ6 is a bi-objective optimization problem")
    
    def evaluate(self, X, unit=False):
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        g = np.sum(X[:, self.n_output-1:] ** 0.1, axis=1)
        Temp = np.tile(g.reshape((-1, 1)), (1, self.n_output-2))
        X[:, 1:self.n_output-2] = (1 + 2 * Temp * X[:, 1:self.n_output-2]) / (2 + 2 * Temp)
        Y = np.tile(1 - g.reshape((-1, 1)), (1, self.n_output)) \
                * np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1)), np.cos(X[:, 0:self.n_output-2] * np.pi / 2))), axis=1)) \
                * np.hstack((np.ones((g.shape[0], 1)), np.sin(X[:, self.n_output-2::-1] * np.pi / 2)))
        
        return Y
    
    def get_optimum(self, N):
        
        R = np.vstack((np.arange(0, 1 + 1/(N-1), 1/(N-1)), np.arange(1, 0 - 1/(N-1), -1/(N-1)))).T
        R = R / np.tile(np.sqrt(np.sum(R**2, axis=1)).reshape((-1, 1)), (1, R.shape[1]))
        R = np.hstack((np.tile(R[:, None, :], (1, self.n_output-2, 1)), R))
        R = R / np.sqrt(2) ** np.tile(np.arange(self.n_output-2, -1, -1), (R.shape[0], 1))
        
        return R
    
    def get_PF(self):
        
        return self.get_optimum(100)
    
    


    

    
        
        

        