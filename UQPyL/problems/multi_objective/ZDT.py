import numpy as np
from typing import Union

from ..problemABC import ProblemABC

##-----------Reference------------------##
# E. Zitzler, K. Deb, and L. Thiele, Comparison of multiobjective
# evolutionary algorithms: Empirical results, Evolutionary computation,
# 2000, 8(2): 173-195.
#--------------------------------------##

class ZDT1(ProblemABC):
    
    name="ZDT1"
    
    def __init__(self, nInput:int =30, nOutput: int=2, 
                    ub: Union[int, float, np.ndarray, list] =1, 
                        lb: Union[int, float, np.ndarray, list] =0):
        
        super().__init__(nInput, nOutput, ub, lb)
        
        if nOutput!=2:
            raise ValueError("ZDT1 is a bi-objective optimization problem")
    
    def objFunc(self, X):
        
        X=self._check_X_2d(X)

        Y = np.zeros((X.shape[0], self.nOutput))
        Y[:,0] = X[:,0]
        g = 1 + 9 * np.mean(X[:, 1:], axis=1)
        h = 1 - np.sqrt(Y[:,0]/g)
        Y[:,1] = g * h
        
        return Y
    
    def getOptimum(self, N=100):
        
        R = np.zeros((N,self.nOutput))
        R[:,0] = np.linspace(0,1,N)
        R[:,1] = 1-np.sqrt(R[:,0])
        
        return R
    
    def get_PF(self):
        
        R=self.getOptimum(100)
        
        return R

class ZDT2(ProblemABC):
    
    name="ZDT2"
    
    def __init__(self, nInput:int =30, nOutput: int=2, 
                    ub: Union[int,float,np.ndarray] =1, 
                        lb: Union[int,float,np.ndarray] =0):
        
        super().__init__(nInput, nOutput, ub, lb)
        
        if nOutput!=2:
            raise ValueError("ZDT2 is a bi-objective optimization problem")
    
    def objFunc(self, X):
        
        X = self._check_X_2d(X)
        
        Y = np.zeros((X.shape[0], self.nOutput))
        Y[:,0] = X[:,0]
        g = 1 + 9 * np.sum(X[:, 1:], axis=1)/(self.nInput-1)
        h = 1-(Y[:,0]/g)**2
        Y[:,1] = g * h
        
        return Y
    
    def getOptimum(self, N=100):
        
        R = np.zeros((N,self.nOutput))
        R[:,0] = np.linspace(0,1,N)
        R[:,1] = 1-(R[:,0])**2
        
        return R
    
    def get_PF(self):
        
        R = self.getOptimum(100)
        
        return R
    
class ZDT3(ProblemABC):
    
    name="ZDT3"
    
    def __init__(self, nInput:int =30, nOutput: int=2, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0):
        
        super().__init__(nInput, nOutput, ub, lb)
        
        if nOutput!=2:
            raise ValueError("ZDT4 is a bi-objective optimization problem")
    
    def objFunc(self, X):
        
        X=self._check_X_2d(X)
        
        Y = np.zeros((X.shape[0], self.nOutput))
        Y[:,0] = X[:,0]
        g = 1 + 9 * np.sum(X[:, 1:], axis=1)/(self.nInput-1)
        h = 1 - np.sqrt(Y[:,0]/g)-(Y[:,0]/g) * np.sin(10*np.pi*Y[:,0])
        Y[:,1] = g * h
        
        return Y
    
    def getOptimum(self, N=100):
        
        from ..utility_functions.NDsort import NDSort
        
        R = np.zeros((N, self.nOutput))
        R[:,0] = np.linspace(0,1,N)
        R[:,1] = 1 - np.sqrt(R[:,0]) - R[:, 0] * np.sin(10 * np.pi * R[:, 0])
        
        FrontNo, _= NDSort(R,1)
        R[FrontNo>1,:] = np.nan
        
        return R
    
    def get_PF(self):
        
        R = self.getOptimum(300)
        
        return R

class ZDT4(ProblemABC):
    
    name="ZDT4"
    
    def __init__(self, nInput:int =30, nOutput: int=2, 
                    ub: Union[int,float,np.ndarray] =1, 
                        lb: Union[int,float,np.ndarray] =0):
       
        super().__init__(nInput, nOutput, ub, lb)
        
        if nOutput!=2:
            raise ValueError("ZDT4 is a bi-objective optimization problem")
    
    def objFunc(self, X):
        
        X=self._check_X_2d(X)
            
        Y = np.zeros((X.shape[0], 2))
        Y[:, 0] = X[:, 0]
        g = 1 + 10 * (X.shape[1] - 1) + np.sum(X[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * X[:, 1:]), axis=1)
        h = 1 - (X[:, 0] / g) ** 0.5
        Y[:, 1] = g * h
        
        return Y
    
    def getOptimum(self, N=100):
        
        R = np.zeros((N, 2))
        R[:, 0] = np.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0] ** 0.5
        
        return R

    def GetPF(self):
        
        R = self.getOptimum(100)
        
        return R

class ZDT6(ProblemABC):
    
    name="ZDT6"
    
    def __init__(self, nInput:int =30, nOutput: int=2, 
                    ub: Union[int,float,np.ndarray] =1, 
                    lb: Union[int,float,np.ndarray] =0):
        
        super().__init__(nInput, nOutput, ub, lb)
        
        if nOutput!=2:
            raise ValueError("ZDT6 is a bi-objective optimization problem")
    
    def objFunc(self, X):
        
        X=self._check_X_2d(X)
        
        Y=np.zeros((X.shape[0], self.nOutput))
        Y[:,0]=1-np.exp(-4*X[:,0])*np.sin(6*np.pi*X[:,0])**6
        g=1+9*np.sum(X[:, 1:], axis=1)/(self.nInput-1)**0.25
        h=1-(Y[:,0]/g)**2
        Y[:,1]=g*h
        
        return Y
    
    def getOptimum(self, N=100):
        
        min=0.280775
        R=np.zeros((N,self.nOutput))
        R[:,0]=np.linspace(min,1,N)
        R[:,1]=1-R[:,0]**2
        
        return R
    
    def get_PF(self):
        
        R=self.getOptimum(100)
        
        return R