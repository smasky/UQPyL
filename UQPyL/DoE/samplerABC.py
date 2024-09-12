import abc
import numpy as np

def decoratorRescale(func):
    def wrapper(self, *args, **kwargs):
        result=func(self, *args, **kwargs)
        if self.problem is not None:
            return self.problem._unit_xInit_transform_to_bound(result)
        else:
            return result
    return wrapper
class Sampler(metaclass=abc.ABCMeta):
    def __init__(self):
        pass
    def __call__(self, nt:int, nx: int) -> np.ndarray:
        return self._generate(nt, nx)
    
    def sample(self, nt:int, nx:int) -> np.ndarray:
        return self._generate(nt, nx)
    
    def rescale_to_problem(self, xInit:np.ndarray):
        if self.problem is not None:
            xInit=self.problem._unit_xInit_transform_to_bound(xInit)
        return xInit
    
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
        nt: the number of sampled points
        nx: the dimensions of decision variables
        
        return:
            ndarry[nt,nx]
        
        '''
        pass

