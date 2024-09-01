import abc
import numpy as np

class Sampler(metaclass=abc.ABCMeta):
    def __init__(self):
        pass
    
    def __call__(self, nt:int, nx: int) -> np.ndarray:
        return self._generate(nt, nx)
    
    def sample(self, nt:int, nx:int) -> np.ndarray:
        return self._generate(nt, nx)
    
    
    def rescale_to_problem(self, X:np.ndarray):
        if self.problem is not None:
            X=self.problem._unit_X_transform_to_bound(X)
        return X
    
    def rescale(self, method):
        def wrapper(X):
            result=method(X)
            
            return self.rescale_to_problem(result)
    
    @abc.abstractmethod
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
        nt: the number of sampled points
        nx: the dimensions of decision variables
        
        return:
            ndarry[nt,nx]
        
        '''
        pass
    
    