import numpy as np
from typing import Optional

from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

class Random(Sampler):
    '''
    Random Design
    
    Method:
    __call__ or sample: Generate a random design
    
    Examples:
        >>> random=RANDOM()
        >>> random(10,10) or random.sample(10,10)
    '''
    def _generate(self,nt: int, nx: int):
        
        H=np.random.random((nt,nx))
        
        return H
    
    @decoratorRescale
    def sample(self, nt: int, nx: Optional[int] = None, problem: Optional[Problem] = None, random_seed: Optional[int] = None):
        '''
        Generate a sample with random values between zero and one 
        '''
        
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
        
        if problem is not None and nx is not None:
            if(problem.nInput!=nx):
                raise ValueError('The input dimensions of the problem and the samples must be the same')
        elif problem is None and nx is None:
            raise ValueError('Either the problem or the input dimensions must be provided')
        
        nx=problem.nInput if problem is not None else nx
        
        return self._generate(nt, nx)