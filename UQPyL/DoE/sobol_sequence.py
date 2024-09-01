import numpy as np
from scipy.stats.qmc import Sobol

from .sampler_ABC import Sampler

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
    
    Examples:
        >>> sobol_seq=Sobol_Sequence(skip_value=128)
        >>> sobol_seq.sample(64, 4)
    '''
    def __init__(self, scramble: bool=True, skip_value: int=0):
        
        super().__init__()
        
        self.scramble=scramble
        self.skip_value=skip_value
        
    @Sampler.rescale
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
        generate the shape of (nt*nx, nx) and numpy array Sobol sequence. 
        '''
        
        sampler=Sobol(d=nx, scramble=self.scramble)
        X=sampler.random(nt+self.skip_value)
        
        return X[self.skip_value:, :]
    
    def sample(self, nt: int, nx: int) -> np.ndarray:
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
        return self._generate(nt, nx)

        
        
        
        
            
        