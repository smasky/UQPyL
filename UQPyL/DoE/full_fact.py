import numpy as np
from typing import Literal

from .sampler_ABC import Sampling

class FFD(Sampling):
    '''
    Full Factorial Design
    
    parameters:
    levels: 2d-array 
        the sampled number of each dimension
    
    return:
    H: 2d-array
    An n-by-samples design matrix between zero and one.
    '''
    
    def __call__(self, levels:np.ndarray) -> np.ndarray:
        
        return self._generate(levels)
    
    def _generate(self,levels: np.ndarray) -> np.ndarray:
        
        levels=np.array(levels,dtype=np.int32).reshape(1,-1)
        
        nx=levels.shape[1]
        nt=np.prod(levels)
        
        H=np.zeros((nt,nx))
        level_repeat = 1
        range_repeat = np.prod(levels)
        for i in range(nx):
            range_repeat = int(range_repeat/levels[0,i])
            lvl = []
            for j in range(levels[0,i]):
                lvl += [j]*level_repeat
            rng = lvl*range_repeat
            level_repeat *= levels[0,i]
            H[:, i] = rng
        
        return H/levels
        
        