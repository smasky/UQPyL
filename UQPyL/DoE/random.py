import numpy as np

from .sampler_ABC import Sampler

class Random(Sampler):
    '''
    Random Design
    
    Method:
    __call__ or sample: Generate a random design
    
    Examples:
        >>> random=RANDOM()
        >>> random(10,10) or random.sample(10,10)
    '''
    def _generate(self,nt: int, nx: int) -> np.ndarray:
        
        H=np.random.random((nt,nx))
        
        return H
    
    def sample(self, nt: int, nx: int) -> np.ndarray:
        '''
        Generate a sample with random values between zero and one 
        '''
        
        return self._generate(nt, nx)