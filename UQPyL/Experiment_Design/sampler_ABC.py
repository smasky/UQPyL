import abc
import numpy as np


class Sampler(metaclass=abc.ABCMeta):
    def __init__(self):
        pass
    
    def __call__(self, nt:int, nx: int) -> np.ndarray:
        return self._generate(nt, nx)
    
    def sample(self, nt:int, nx:int) -> np.ndarray:
        return self._generate(nt, nx)
    
    @abc.abstractmethod
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
        nt: the number of sampled points
        nx: the dimensions of decision variables
        
        return:
            ndarry[nt,nx]
        
        '''
        pass
    
    