import numpy as np

from .sampler_ABC import Sampling

class RANDOM(Sampling):
    '''
    random design
    '''
    def _generate(self,nt: int, nx: int) -> np.ndarray:
        H=np.random.random((nt,nx))
        return H