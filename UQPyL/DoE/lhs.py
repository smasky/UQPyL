import numpy as np
from typing import Literal

from .sampler_ABC import Sampling
from ._lhs import _lhs_classic, _lhs_centered, _lhs_correlate, _lhs_maximin, _lhs_centered_maximin

Criterion=Literal['classic','center','maximin','center_maximin','correlation']
LHS_METHOD={'classic': _lhs_classic, 'center': _lhs_centered, 'maximin': _lhs_maximin,
            'center_maximin': _lhs_centered_maximin, 'correlation': _lhs_correlate}

class LHS(Sampling):
    '''
    Latin-hypercube design
    
    Parameters
    _________________
    criterion : str
        Allowable values are "classic","center", "maximin", "center_maximin", 
        and "correlation". (Default: classic)
    iterations : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).
    
    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.
    '''
    def __init__(self, criterion: Criterion='classic', iterations: int=5)-> None:
        self.criterion=criterion
        self.iterations=iterations
        
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        
        Sampling_method=LHS_METHOD[self.criterion]
        if self.criterion in ['maximin', 'center_maximin', 'correlation']:
            return Sampling_method(nt, nx, self.iterations)
        else:
            return Sampling_method(nt, nx)