import numpy as np
import itertools
from typing import Union

from ..base import ProblemABC

##----------------Reference-------------------#
# K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, Scalable test problems
# for evolutionary multiobjective optimization, Evolutionary multiobjective
# Optimization. Theoretical Advances and Applications, 2005, 105-145.
##--------------------------------------------#
class DTLZ1(ProblemABC):
    '''
    Multi-Objective problem named DTLZ1 of the DTLZ suit.
    
    Methods:
    objFunc: Returns the function value of the problem if provide the X.
    getPF: Returns the Pareto Front of the problem.
    getOptimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    nInput: int
        Dimension of the problem.
    nOutput: int
        Number of objective functions.
    ub: Union[int,float,np.ndarray]
        Upper bound of the problem.
    lb: Union[int,float,np.ndarray]
        Lower bound of the problem.
    disc_var: list
        Discrete variables of the problem.
    cont_var: list
        Continuous variables of the problem.
    '''
    
    name="DTLZ1"
    
    def __init__(self, nInput:int = 30, nOutput: int = 3, 
                    ub: Union[int,float,np.ndarray] = 1, 
                        lb: Union[int,float,np.ndarray] = 0):
        
        super().__init__(nInput, nOutput, ub, lb)
    
    def objFunc(self, X):
        '''
        Perform the evaluation of the input variables X
        
        Parameters:
        X: np.ndarray(2d-array)
            input variables
                    
        Returns:
        Y: np.ndarray(2d-array)
            the outputs of the problem.
        '''
        
        X = self._check_X_2d(X)
        
        n_samples = X.shape[0]
        
        g = 100 * (self.nInput - self.nOutput + 1 + \
               np.sum((X[:, self.nOutput-1:] - 0.5) ** 2 - \
                      np.cos(20. * np.pi * (X[:, self.nOutput-1:] - 0.5)), axis=1))
        
        Y = np.zeros((n_samples, self.nOutput))
        
        for i in range(self.nOutput):
           
            if i < self.nOutput - 1:
                prefix_prod = np.prod(X[:, :self.nOutput-1-i], axis=1)
            else:
                prefix_prod = np.ones(n_samples)
            
            if i == 0:
                suffix_term = np.ones(n_samples)
            else:
                suffix_term = 1 - X[:, self.nOutput-1-i]
          
            Y[:, i] = 0.5 * (1 + g) * prefix_prod * suffix_term
        
        return Y
    
    def getOptimum(self, n):
        
        '''
        Return the optimum of the problem.
        '''
        
        from ..util.uniformPoint import uniformPoint
        
        R, _ = uniformPoint(N, self.nOutput)
        
        R = R * 0.5
        
        R = np.maximum(R, 0)
        
        return R

    def getPF(self):
        '''
        Return the pareto front of the problem.
        '''
        
        if self.nOutput == 3:
            res = 201  
            s, t = np.meshgrid(np.linspace(0.0, 1.0, res),
                            np.linspace(0.0, 1.0, res))
        
            f1 = 0.5 * s * (1.0 - t)
            f2 = 0.5 * (1.0 - s) * (1.0 - t)
            f3 = 0.5 * t

            return (f1, f2, f3)

class DTLZ2(ProblemABC):
    '''
    Multi-Objective problem named DTLZ2 of the DTLZ suit.
    
    Methods:
    objFunc: Returns the function value of the problem if provide the X.
    getPF: Returns the Pareto Front of the problem.
    getOptimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    nInput: int
        Dimension of the problem.
    nOutput: int
        Number of objective functions.
    ub: Union[int,float,np.ndarray]
        Upper bound of the problem.
    lb: Union[int,float,np.ndarray]
        Lower bound of the problem.
    disc_var: list
        Discrete variables of the problem.
    cont_var: list
        Continuous variables of the problem.
    '''
    
    name="DTLZ2"
    
    def __init__(self, nInput:int =30, nOutput: int=3, 
                    ub: Union[int,float,np.ndarray] =1, 
                        lb: Union[int,float,np.ndarray] =0):
           
        super().__init__(nInput, nOutput, ub, lb)
        
        if nOutput!=3:
            raise ValueError("DTLZ2 is a three-objective optimization problem")
    
    def objFunc(self, X):
        '''
        Perform the evaluation of the input variables X
        
        Parameters:
        X: np.ndarray(2d-array)
            input variables
        unit: bool
            if True, the input variables will be transformed to the zero-one bound of the problem.
        
        Returns:
        Y: np.ndarray(2d-array)
            the outputs of the problem.
        '''
        X=self._check_X_2d(X)
        
        g = np.sum((X[:, self.nOutput:] - 0.5) ** 2, axis=1)
        ones_col = np.ones((g.shape[0], 1))
        cos_prod = np.cos(X[:, :self.nOutput-1] * np.pi / 2)
        sin_vals = np.sin(X[:, self.nOutput-2::-1] * np.pi / 2)
        
        cumprod_part = np.cumprod(np.hstack([ones_col, cos_prod]), axis=1)
        Y = np.tile(1 + g, (self.nOutput, 1)).T * np.fliplr(cumprod_part) * np.hstack([ones_col, sin_vals])
        
        return Y
    
    def getOptimum(self, N):
        '''
        Return the optimum of the problem.
        '''
        from ..util.uniformPoint import uniformPoint
        R, _ = uniformPoint(N, self.nOutput)
        R = R / np.tile(np.sqrt(np.sum(R ** 2, axis=1)).reshape(-1, 1), (1, self.nOutput))
        
        return R
    
    def getPF(self):
        '''
        Return the pareto front of the problem.
        '''
        
        if self.nOutput == 3:
            res = 101 
            theta = np.linspace(0.0, np.pi/2, res) 
            phi   = np.linspace(0.0, np.pi/2, res) 
            T, P = np.meshgrid(theta, phi)

            f1 = np.cos(T) * np.cos(P)
            f2 = np.cos(T) * np.sin(P)
            f3 = np.sin(T)

            return (f1, f2, f3)
    
class DTLZ3(ProblemABC):
    '''
    Multi-Objective problem named DTLZ3 of the DTLZ suit.
    
    Methods:
    objFunc: Returns the function value of the problem if provide the X.
    getPF: Returns the Pareto Front of the problem.
    getOptimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    nInput: int
        Dimension of the problem.
    nOutput: int
        Number of objective functions.
    ub: Union[int,float,np.ndarray]
        Upper bound of the problem.
    lb: Union[int,float,np.ndarray]
        Lower bound of the problem.
    disc_var: list
        Discrete variables of the problem.
    cont_var: list
        Continuous variables of the problem.
    '''
    
    name="DTLZ3"
    
    def __init__(self, nInput:int =30, nOutput: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0):
        
        super().__init__(nInput, nOutput, ub, lb)
         
        if nOutput!=3:
            raise ValueError("DTLZ3 is a three-objective optimization problem")
    
    def objFunc(self, X):
        '''
        Perform the evaluation of the input variables X
        
        Parameters:
        X: np.ndarray(2d-array)
            input variables
        unit: bool
            if True, the input variables will be transformed to the zero-one bound of the problem.
        
        Returns:
        Y: np.ndarray(2d-array)
            the outputs of the problem.
        '''
        X=self._check_X_2d(X)
        
        g = 100 * (self.nInput - self.nOutput + 1 + np.sum((X[:, self.nOutput-1:] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, self.nOutput-1:] - 0.5)), axis=1))
        Y = (1 + g[:, None]) * np.fliplr(np.cumprod(np.hstack([np.ones((X.shape[0], 1)), np.cos(X[:, :self.nOutput-1] * np.pi / 2)]), axis=1)) * np.hstack([np.ones((X.shape[0], 1)), np.sin(X[:, self.nOutput-2::-1] * np.pi / 2)])
        return Y
    
    def getOptimum(self, N):
        '''
        Return the optimum of the problem.
        '''
        from ..util.uniformPoint import uniformPoint
        R, _ =uniformPoint(N, self.nOutput)
        R /= np.sqrt(np.sum(R**2, axis=1))[:, np.newaxis]
        
        return R
    
    def getPF(self):
        '''
        Return the pareto front of the problem.
        '''
        
        if self.nOutput == 3:
            res = 101  
            theta = np.linspace(0.0, np.pi/2, res) 
            phi   = np.linspace(0.0, np.pi/2, res) 
            T, P = np.meshgrid(theta, phi)

            f1 = np.cos(T) * np.cos(P)
            f2 = np.cos(T) * np.sin(P)
            f3 = np.sin(T)

            return (f1, f2, f3)

class DTLZ4(ProblemABC):
    '''
    Multi-Objective problem named DTLZ4 of the DTLZ suit.
    
    Methods:
    objFunc: Returns the function value of the problem if provide the X.
    getPF: Returns the Pareto Front of the problem.
    getOptimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    nInput: int
        Dimension of the problem.
    nOutput: int
        Number of objective functions.
    ub: Union[int,float,np.ndarray]
        Upper bound of the problem.
    lb: Union[int,float,np.ndarray]
        Lower bound of the problem.
    disc_var: list
        Discrete variables of the problem.
    cont_var: list
        Continuous variables of the problem.
    '''
    
    name="DTLZ4"
    
    def __init__(self, nInput:int =30, nOutput: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0):
        
        super().__init__(nInput, nOutput, ub, lb)
        
        if nOutput!=3:
            raise ValueError("DTLZ4 is a three-objective optimization problem")
    
    def objFunc(self, X):
        '''
        Perform the evaluation of the input variables X
        
        Parameters:
        X: np.ndarray(2d-array)
            input variables
        unit: bool
            if True, the input variables will be transformed to the zero-one bound of the problem.
        
        Returns:
        Y: np.ndarray(2d-array)
            the outputs of the problem.
        '''
        X=self._check_X_2d(X)
            
        X[:, :self.nOutput-1] = np.power(X[:, :self.nOutput-1], 100)
        g = np.sum(np.power(X[:, self.nOutput-1:] - 0.5, 2), axis=1)
        Y = np.tile(1 + g[:, None], (1, self.nOutput)) \
            * np.fliplr(np.cumprod(np.hstack([np.ones((g.shape[0], 1)), np.cos(X[:, :self.nOutput-1] * np.pi / 2)]), axis=1)) \
                * np.hstack([np.ones((g.shape[0], 1)), np.sin(X[:, self.nOutput-2::-1] * np.pi / 2)])
        
        return Y
    
    def getOptimum(self, N):
        '''
        Return the optimum of the problem.
        '''
        from ..util.uniformPoint import uniformPoint
        R, _ = uniformPoint(N, self.nOutput)
        R /= np.sqrt(np.sum(R**2, axis=1))[:, np.newaxis]
        return R

    def getPF(self):
        '''
        Return the pareto front of the problem.
        '''
        
        if self.nOutput == 3:
        
            res = 101 
            theta = np.linspace(0.0, np.pi/2, res)  
            phi   = np.linspace(0.0, np.pi/2, res)  
            T, P = np.meshgrid(theta, phi)

            f1 = np.cos(T) * np.cos(P)
            f2 = np.cos(T) * np.sin(P)
            f3 = np.sin(T)

            return (f1, f2, f3)
        
class DTLZ5(ProblemABC):
    '''
    Multi-Objective problem named DTLZ5 of the DTLZ suit.
    
    Methods:
    objFunc: Returns the function value of the problem if provide the X.
    getPF: Returns the Pareto Front of the problem.
    getOptimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    nInput: int
        Dimension of the problem.
    nOutput: int
        Number of objective functions.
    ub: Union[int,float,np.ndarray]
        Upper bound of the problem.
    lb: Union[int,float,np.ndarray]
        Lower bound of the problem.
    disc_var: list
        Discrete variables of the problem.
    cont_var: list
        Continuous variables of the problem.
    '''
    
    name="DTLZ5"
    
    def __init__(self, nInput:int =30, nOutput: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0):
        
        super().__init__(nInput, nOutput, ub, lb)
         
        if nOutput!=3:
            raise ValueError("DTLZ5 is a three-objective optimization problem")
    
    def objFunc(self, X):
        '''
        Perform the evaluation of the input variables X
        
        Parameters:
        X: np.ndarray(2d-array)
            input variables
        unit: bool
            if True, the input variables will be transformed to the zero-one bound of the problem.
        
        Returns:
        Y: np.ndarray(2d-array)
            the outputs of the problem.
        '''
        X=self._check_X_2d(X)

        g = np.sum((X[:, self.nOutput-1:] - 0.5)**2, axis=1)
        temp = np.tile(g[:, None], (1, self.nOutput-2))
        X[:, 1:self.nOutput-1] = (1 + 2 * temp * X[:, 1:self.nOutput-1]) / (2 + 2 * temp)
        Y = np.tile(1 + g[:, None], (1, self.nOutput)) \
            * np.fliplr(np.cumprod(np.hstack([np.ones((g.shape[0], 1)), np.cos(X[:, :self.nOutput-1] * np.pi / 2)]), axis=1)) \
                * np.hstack([np.ones((g.shape[0], 1)), np.sin(X[:, self.nOutput-2::-1] * np.pi / 2)])
        return Y
    
    def getOptimum(self, N):
        '''
        Return the optimum of the problem.
        '''
       
        R = np.column_stack((np.linspace(0, 1, N), np.linspace(1, 0, N)))  
     
        R = R / np.sqrt(np.sum(R**2, axis=1, keepdims=True))  

      
        R_extended = np.hstack((R[:, np.zeros(self.nOutput-2, dtype=int)], R)) 

        scaling_factors = np.sqrt(2) ** np.array([self.nOutput-2] + list(range(self.nOutput-2, -1, -1)))  
        R = R_extended / scaling_factors  
        
        return R

    def getPF(self):
        '''
        Return the pareto front of the problem.
        '''
        
        if self.nOutput == 3:
        
            N = 200  
            theta1 = np.linspace(0.0, np.pi/2, N)   
            c = np.cos(theta1)
            s = np.sin(theta1)
            coef = 1.0 / np.sqrt(2.0)             

            f1 = (coef * c)[:, np.newaxis]  
            f2 = (coef * c)[:, np.newaxis]   
            f3 = (s)[:, np.newaxis]
                
            return (f1, f2, f3)

class DTLZ6(ProblemABC):
    '''
    Multi-Objective problem named DTLZ6 of the DTLZ suit.
    
    Methods:
    objFunc: Returns the function value of the problem if provide the X.
    getPF: Returns the Pareto Front of the problem.
    getOptimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    nInput: int
        Dimension of the problem.
    nOutput: int
        Number of objective functions.
    ub: Union[int,float,np.ndarray]
        Upper bound of the problem.
    lb: Union[int,float,np.ndarray]
        Lower bound of the problem.
    disc_var: list
        Discrete variables of the problem.
    cont_var: list
        Continuous variables of the problem.
    '''
    
    name="DTLZ6"
    
    def __init__(self, nInput:int =30, nOutput: int=3, 
                    ub: Union[int,float,np.ndarray] =1, 
                        lb: Union[int,float,np.ndarray] =0):
        
        super().__init__(nInput, nOutput, ub, lb)
        
        if nOutput!=3:
            raise ValueError("DTLZ6 is a three-objective optimization problem")
    
    def objFunc(self, X):
        '''
        Perform the evaluation of the input variables X
        
        Parameters:
        X: np.ndarray(2d-array)
            input variables
        unit: bool
            if True, the input variables will be transformed to the zero-one bound of the problem.
        
        Returns:
        Y: np.ndarray(2d-array)
            the outputs of the problem.
        '''
        X=self._check_X_2d(X)
        
        g = np.sum(X[:, self.nOutput:]**0.1, axis=1)
        Temp = np.tile(g, (self.nOutput-2, 1)).T
        X[:, 1:self.nOutput-1] = (1 + 2 * Temp * X[:, 1:self.nOutput-1]) / (2 + 2 * Temp)
        Y = np.tile(1 + g, (self.nOutput, 1)).T \
            * np.fliplr(np.cumprod(np.column_stack([np.ones(g.shape), np.cos(X[:, :self.nOutput-1] * np.pi / 2)]), axis=1)) \
                * np.column_stack([np.ones(g.shape), np.sin(X[:, self.nOutput-2::-1] * np.pi / 2)])
        
        return Y
    
    def getOptimum(self, N):
        '''
        Return the optimum of the problem.
        '''

        R = np.array([np.linspace(0, 1, N), np.linspace(1, 0, N)]).T
        R = R / np.sqrt(np.sum(R**2, axis=1, keepdims=True))
        R = np.hstack([R[:, [0]] * np.ones((1, self.nOutput - 2)), R])
        scale_factors = np.sqrt(2) ** np.array([self.nOutput - 2] + list(range(self.nOutput - 2, -1, -1)))
        R = R / scale_factors

        return R
    
    def getPF(self):
        '''
        Return the pareto front of the problem.
        '''
        
        if self.nOutput == 3:
            N = 200  
            theta1 = np.linspace(0.0, np.pi/2, N)   
            c = np.cos(theta1)
            s = np.sin(theta1)
            coef = 1.0 / np.sqrt(2.0)             

            f1 = (coef * c)[:, np.newaxis]  
            f2 = (coef * c)[:, np.newaxis]   
            f3 = (s)[:, np.newaxis]
                
            return (f1, f2, f3)

class DTLZ7(ProblemABC):
    '''
    Multi-Objective problem named DTLZ7 of the DTLZ suit.
    
    Methods:
    objFunc: Returns the function value of the problem if provide the X.
    getPF: Returns the Pareto Front of the problem.
    getOptimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    nInput: int
        Dimension of the problem.
    nOutput: int
        Number of objective functions.
    ub: Union[int,float,np.ndarray]
        Upper bound of the problem.
    lb: Union[int,float,np.ndarray]
        Lower bound of the problem.
    disc_var: list
        Discrete variables of the problem.
    cont_var: list
        Continuous variables of the problem.
    '''
    
    name="DTLZ7"
    
    def __init__(self, nInput:int = 30, nOutput: int = 3, 
                    ub: Union[int,float,np.ndarray] = 1, 
                        lb: Union[int,float,np.ndarray] = 0):
        
        super().__init__(nInput, nOutput, ub, lb)
        
        if nOutput != 3:
            raise ValueError("DTLZ6 is a three-objective optimization problem")
        
    def objFunc(self, X):
        '''
        Perform the evaluation of the input variables X
        
        Parameters:
        X: np.ndarray(2d-array)
            input variables
        unit: bool
            if True, the input variables will be transformed to the zero-one bound of the problem.
        
        Returns:
        Y: np.ndarray(2d-array)
            the outputs of the problem.
        '''
        X=self._check_X_2d(X)
        
        g = 1 + 9 * np.mean(X[:, self.nOutput:], axis=1, keepdims=True)

        Y = np.hstack([
            X[:, :self.nOutput-1], 
            (1 + g) * (self.nOutput - np.sum(
                X[:, :self.nOutput-1] / (1 + g) * (1 + np.sin(3 * np.pi * X[:, :self.nOutput-1])),
                axis=1,
                keepdims=True
            ))
        ])
        
        return Y
    
    def getOptimum(self, N):
        '''
        Return the optimum of the problem.
        '''
        from ..util.uniformPoint import uniformPoint
        
        interval = [0, 0.251412, 0.631627, 0.859401]
        
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        
        X, _ = uniformPoint(N, self.nOutput-1, 'grid')
        X[X <= median] = X[X <= median] * (interval[1] - interval[0]) / median + interval[0]
        X[X > median] = (X[X > median] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
        R = np.hstack((X, 2 * (self.nOutput - np.sum(X / 2 * (1 + np.sin(3 * np.pi * X)), axis=1)).reshape(-1, 1)))
        
        return R
    
    def getPF(self):
        '''
        Return the pareto front of the problem.
        '''
        
        if self.nOutput == 3:
            
            from ..util.non_dominated_sort import NDSort
            
            f1, f2 = np.meshgrid(np.linspace(0.0, 1.0, 51),
                            np.linspace(0.0, 1.0, 51))
            
            f3 = 2 * (3
                - 0.5 * f1 * (1 + np.sin(3 * np.pi * f1))
                - 0.5 * f2 * (1 + np.sin(3 * np.pi * f2)))
            
            F = np.stack([f1.ravel(), f2.ravel(), f3.ravel()], axis=1)
            
            frontNo, _ = NDSort(F)
            
            nd = frontNo == 1
            nd = nd.reshape(f3.shape)
            
            f3_plot = f3.copy()
            f3_plot[~nd] = np.nan
            
            return (f1, f2, f3_plot)