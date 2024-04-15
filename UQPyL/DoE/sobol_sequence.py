import numpy as np
from scipy.stats.qmc import Sobol

from .sampler_ABC import Sampler

class Sobol_Sequence(Sampler):
    '''
    Sobol Sequence
    
    Methods:
    __call__ or sample: generate the shape of (nt*nx, nx) and numpy array Sobol sequence. 
    
    '''
    def __init__(self):
        
        super().__init__()
    
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
        generate the shape of (nt*nx, nx) and numpy array Sobol sequence. 
        '''
        
        return Sobol(d=nx).random(nt)
    
    def sample(self, nt: int, nx: int) -> np.ndarray:
        '''
        generate the shape of (nt*nx, nx) and numpy array Sobol sequence. 
        
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

        
        
        
        
            
        