import numpy as np
import itertools
from typing import Union

from .problem_ABC import ProblemABC

##----------------Reference-------------------#
# K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, Scalable test problems
# for evolutionary multiobjective optimization, Evolutionary multiobjective
# Optimization. Theoretical Advances and Applications, 2005, 105-145.
##--------------------------------------------#
class DTLZ1(ProblemABC):
    '''
    Multi-Objective problem named DTLZ1 of the DTLZ suit.
    
    Methods:
    evaluate: Returns the function value of the problem if provide the X.
    get_PF: Returns the Pareto Front of the problem.
    get_optimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    n_input: int
        Dimension of the problem.
    n_output: int
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
    def __init__(self, n_input:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ1 is a three-objective optimization problem")
    
    def evaluate(self, X, unit=False):
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
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
            
        g = 100 * (self.n_input - self.n_output + 1 + \
           np.sum((X[:, self.n_output:] - 0.5) ** 2 - \
                  np.cos(20. * np.pi * (X[:, self.n_output:] - 0.5)), axis=1))
        
        Y = 0.5 * np.tile(1 + g, (1, self.n_output)) \
            * np.fliplr(np.cumprod(np.hstack([np.ones((X.shape[0], 1)), X[:, :self.n_output - 1]]), axis=1)) \
            * np.hstack([np.ones((X.shape[0], 1)), 1 - X[:, self.n_output - 1::-1]])
        
        return Y
    
 
    
    def get_optimum(self, N):
        '''
        Return the optimum of the problem.
        '''
        from .utility_functions._uniformPoint import uniformPoint
        R,_= uniformPoint(N, self.n_output)
        R=R/2
        
        return R

    def get_PF(self):
        '''
        Return the pareto front of the problem.
        '''
        #TODO 
        a = np.linspace(0, 1, 10).reshape(-1, 1)
        R = [a.dot(a.T)/2, a.dot((1 - a.T))/2, (1 - a).dot(np.ones(a.T.shape))/2]
        Y = np.array(list(itertools.product(R[0], R[1], R[2])))
          
        return Y

class DTLZ2(ProblemABC):
    '''
    Multi-Objective problem named DTLZ2 of the DTLZ suit.
    
    Methods:
    evaluate: Returns the function value of the problem if provide the X.
    get_PF: Returns the Pareto Front of the problem.
    get_optimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    n_input: int
        Dimension of the problem.
    n_output: int
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
    def __init__(self, n_input:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ2 is a three-objective optimization problem")
    
    def evaluate(self, X, unit=False):
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
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        g = np.sum((X[:, self.n_output:] - 0.5) ** 2, axis=1)
        Y = np.tile(1 + g, (1, self.n_output)) \
            * np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1)), np.cos(X[:, :self.n_output - 1] * np.pi / 2))), axis=1)) \
            * np.hstack((np.ones((g.shape[0], 1)), np.sin(X[:, self.n_output - 1::-1] * np.pi / 2)))
        
        return Y
    
    def get_optimum(self, N):
        '''
        Return the optimum of the problem.
        '''
        from .utility_functions._uniformPoint import uniformPoint
        R, _ = uniformPoint(N, self.n_output)
        R = R / np.tile(np.sqrt(np.sum(R ** 2, axis=1)).reshape(-1, 1), (1, self.n_output))
        
        return R
    
    def get_PF(self):
        '''
        Return the pareto front of the problem.
        '''
        a = np.linspace(0, np.pi / 2, 10).reshape(-1, 1)
        R = [np.sin(a) * np.cos(a.T), np.sin(a) * np.sin(a.T), np.cos(a) * np.ones(a.shape).T]
        Y = np.array(list(itertools.product(R[0], R[1], R[2])))
        #TODO 
        return Y

class DTLZ3(ProblemABC):
    '''
    Multi-Objective problem named DTLZ3 of the DTLZ suit.
    
    Methods:
    evaluate: Returns the function value of the problem if provide the X.
    get_PF: Returns the Pareto Front of the problem.
    get_optimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    n_input: int
        Dimension of the problem.
    n_output: int
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
    def __init__(self, n_input:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ3 is a three-objective optimization problem")
    
    def evaluate(self, X, unit=False):
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
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        g = 100 * (self.n_input - self.n_output + 1 + np.sum((X[:, self.n_output:] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, self.n_output:] - 0.5)), axis=1))
        Y = np.tile(1 + g, (1, self.n_output)) * np.fliplr(np.cumprod(np.hstack([np.ones((X.shape[0], 1)), np.cos(X[:, :self.n_output - 1] * np.pi / 2)]), axis=1)) * np.hstack([np.ones((X.shape[0], 1)), np.sin(X[:, self.n_output - 1::-1] * np.pi / 2)])
        return Y
    
    def get_optimum(self, N):
        '''
        Return the optimum of the problem.
        '''
        from .utility_functions._uniformPoint import uniformPoint
        R, _ =uniformPoint(N, self.n_output)
        R /= np.sqrt(np.sum(R**2, axis=1))[:, np.newaxis]
        
        return R
    
    def get_PF(self):
        '''
        Return the pareto front of the problem.
        '''
        a = np.linspace(0, np.pi / 2, 10)
        R = [np.sin(a) * np.cos(a), np.sin(a) * np.sin(a), np.cos(a) * np.ones(a.shape)]
        # Y = np.array(list(itertools.product(R[0], R[1], R[2])))
        #TODO 
        return R

class DTLZ4(ProblemABC):
    '''
    Multi-Objective problem named DTLZ4 of the DTLZ suit.
    
    Methods:
    evaluate: Returns the function value of the problem if provide the X.
    get_PF: Returns the Pareto Front of the problem.
    get_optimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    n_input: int
        Dimension of the problem.
    n_output: int
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
    def __init__(self, n_input:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ4 is a three-objective optimization problem")
    
    def evaluate(self, X, unit=False):
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
        '''
        Return the optimum of the problem.
        '''
        from .utility_functions._uniformPoint import uniformPoint
        R, _ = uniformPoint(N, self.n_output)
        R /= np.sqrt(np.sum(R**2, axis=1))[:, np.newaxis]
        return R

    def get_PF(self):
        '''
        Return the pareto front of the problem.
        '''
        a = np.linspace(0, np.pi/2, 10)
        R = [np.sin(a) * np.cos(a), np.sin(a) * np.sin(a), np.cos(a) * np.ones_like(a)]
        Y = np.array(list(itertools.product(R[0], R[1], R[2])))
        #TODO 
        return Y
    
class DTLZ5(ProblemABC):
    '''
    Multi-Objective problem named DTLZ5 of the DTLZ suit.
    
    Methods:
    evaluate: Returns the function value of the problem if provide the X.
    get_PF: Returns the Pareto Front of the problem.
    get_optimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    n_input: int
        Dimension of the problem.
    n_output: int
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
    def __init__(self, n_input:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ5 is a three-objective optimization problem")
    
    def evaluate(self, X, unit=False):
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
        '''
        Return the optimum of the problem.
        '''
        R = np.vstack((np.linspace(0, 1, N), np.linspace(1, 0, N))).T
        # 规范化这些点，使其在目标空间中的范数为 1
        R /= np.linalg.norm(R, axis=1, keepn_inputs=True)
        # 扩展到更高维度，重复 R 的第一列 obj.M-2 次
        R = np.hstack([np.tile(R[:, [0]], (1, self.n_output-2)), R])
        # 计算规范化的权重因子
        divisors = np.power(np.sqrt(2), np.tile([self.n_output-2] + list(range(self.n_output-2, -1, -1)), (R.shape[0], 1)))
        R /= divisors
        return R

    def get_pf(self):
        '''
        Return the pareto front of the problem.
        '''
        #TODO 
        return self.get_optimum(100)

class DTLZ6(ProblemABC):
    '''
    Multi-Objective problem named DTLZ6 of the DTLZ suit.
    
    Methods:
    evaluate: Returns the function value of the problem if provide the X.
    get_PF: Returns the Pareto Front of the problem.
    get_optimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    n_input: int
        Dimension of the problem.
    n_output: int
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
    def __init__(self, n_input:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ6 is a three-objective optimization problem")
    
    def evaluate(self, X, unit=False):
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
        '''
        Return the optimum of the problem.
        '''
        R = np.array([np.linspace(0, 1, N), np.linspace(1, 0, N)]).T
        R = R / np.sqrt(np.sum(R**2, axis=1)).reshape(-1, 1)
        R = np.hstack([R[:, self.n_output-3:self.n_output], R])
        R = R / np.power(np.sqrt(2), np.tile([self.n_output-2] + list(range(self.n_output-2, -1, -1)), (R.shape[0], 1)))

        return R
    
    def get_PF(self):
        '''
        Return the pareto front of the problem.
        '''
        #TODO 
        return self.get_optimum(100)

class DTLZ7(ProblemABC):
    '''
    Multi-Objective problem named DTLZ7 of the DTLZ suit.
    
    Methods:
    evaluate: Returns the function value of the problem if provide the X.
    get_PF: Returns the Pareto Front of the problem.
    get_optimum: Returns the Pareto Optimum of the problem.
    
    Attributes:
    n_input: int
        Dimension of the problem.
    n_output: int
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
    def __init__(self, n_input:int =30, n_output: int=3, ub: Union[int,float,np.ndarray] =1, lb: Union[int,float,np.ndarray] =0,disc_var=None,cont_var=None) -> None:
        
        self.n_input=n_input
        self.n_output=n_output
        self._set_ub_lb(ub,lb)
        
        self.disc_var=disc_var
        self.cont_var=cont_var
        if n_output!=3:
            raise ValueError("DTLZ6 is a three-objective optimization problem")
        
    def evaluate(self, X, unit=False):
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
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        
        g = 1 + 9 * np.mean(X[:, self.n_output:], axis=1)
        Y = np.zeros((X.shape[0], self.n_output))
        Y[:, :self.n_output-1] = X[:, :self.n_output-1]
        Y[:, self.n_output-1] = (1 + g) * (self.n_output - np.sum(Y[:, :self.n_output-1] / (1 + np.repeat(g, self.n_output-1)) * (1 + np.sin(3*np.pi*Y[:, :self.n_output-1])), axis=1))
    
    def get_optimum(self, N):
        '''
        Return the optimum of the problem.
        '''
        from .utility_functions._uniformPoint import uniformPoint
        interval = [0, 0.251412, 0.631627, 0.859401]
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        
        X, _ = uniformPoint(N, self.n_output-1, 'grid')
        X[X <= median] = X[X <= median] * (interval[1] - interval[0]) / median + interval[0]
        X[X > median] = (X[X > median] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
        R = np.hstack((X, 2 * (self.n_output - np.sum(X / 2 * (1 + np.sin(3 * np.pi * X)), axis=1)).reshape(-1, 1)))
        
        return R
    
    def get_PF(self):
        '''
        Return the pareto front of the problem.
        '''
        #TODO 
        pass


    

    
        
        

        