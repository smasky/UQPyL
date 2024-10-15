import numpy as np
from scipy.stats.qmc import Sobol
from typing import Optional

from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

class Sobol_Sequence(Sampler):
    '''
    Sobol Sequence
    ------------------------------------------------
    Parameters:
    scramble: bool default=True
        the switch to scramble the sequence or not
    skip_value: int default=0
        the number of skipped points for Sobol sequence
    
    Methods:
    __call__ or sample: generate the shape of (nt*nx, nx) and numpy array Sobol sequence. 
    
    '''
    def __init__(self, scramble: bool=True, skipValue: int=0):
        
        super().__init__()
        
        self.scramble=scramble
        self.skipValue=skipValue
        
    def _generate(self, nt: int, nx: int):
        '''
        generate the shape of (nt*nx, nx) and numpy array Sobol sequence. 
        '''
        
        sampler=Sobol(d=nx, scramble=self.scramble)
        xInit=sampler.random(nt+self.skipValue)
        
        return xInit[self.skipValue:, :]
    
    @decoratorRescale
    def sample(self, nt: int, nx: Optional[int] = None, problem: Optional[Problem] = None, random_seed: Optional[int] = None) -> np.ndarray:
        '''
        generate the shape of (nt, nx) and numpy array Sobol sequence. 
        
        Parameters 
        nt: int
            the number of sampled points
        nx: int
            the input dimensions of sampled points
            
        Returns:
        H: 2d-array
            An n-by-samples design matrix that has been normalized so factor values
            are uniformly spaced between zero and one.  
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