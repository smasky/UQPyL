import abc
import numpy as np

from ..problem import ProblemABC as Problem

class Sampler(metaclass = abc.ABCMeta):
    
    def __init__(self):
        
        pass
    
    def _generate(self, nt: int, nx: int, seed = None):
        '''
        nt: the number of sampled points
        nx: the dimensions of decision variables
        
        return:
            ndarry[nt,nx]
        '''
        
        pass

