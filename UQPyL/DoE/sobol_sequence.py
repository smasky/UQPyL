import numpy as np
from scipy.stats.qmc import Sobol

from .sampler_ABC import Sampler

class Sobol_Sequence(Sampler):
    def __init__(self):
        super().__init__()
    
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
            generate the shape of (nt*nx, nx) and numpy array Sobol sequence. 
        '''
        return Sobol(d=nx).random(nt)

        
        
        
        
            
        