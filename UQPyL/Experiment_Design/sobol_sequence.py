import numpy as np
from .sampler_ABC import Sampler
from scipy.stats import qmc


class Sobol_Sample(Sampler):
    def __init__(self):
        super().__init__()
    
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
            generate the shape of (nt*nx, nx) and numpy array Sobol sequence. 
        '''
        
        
        
            
        