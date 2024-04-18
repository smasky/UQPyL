import numpy as np
from typing import Literal

from .sampler_ABC import Sampler
from ._lhs import _lhs_classic, _lhs_centered, _lhs_correlate, _lhs_maximin, _lhs_centered_maximin

Criterion=Literal['classic','center','maximin','center_maximin','correlation']
LHS_METHOD={'classic': _lhs_classic, 'center': _lhs_centered, 'maximin': _lhs_maximin,
            'center_maximin': _lhs_centered_maximin, 'correlation': _lhs_correlate}

class LHS(Sampler):
    '''
    Latin-hypercube design
    
    Parameters:
    criterion : str
        Allowable values are "classic", "center", "maximin", "center_maximin", 
        and "correlation". (Default: classic)
        
    iterations : int
        The number of iterations in the maximin, center_maximin and correlations methods
        (Default: 5).
    
    Methods:
    __call__ or sample: Generate a Latin-hypercube design
        
    Examples:
        >>>lhs=LHS('classic')
        >>>samples=lhs(5,10) or samples=lhs.sample(5,10)
    
    '''
    def __init__(self, criterion: Criterion='classic', iterations: int=5)-> None:
        self.criterion=criterion
        self.iterations=iterations
        
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
        Generate a Latin-hypercube design
        
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
        Sampling_method=LHS_METHOD[self.criterion]
        if self.criterion in ['maximin', 'center_maximin', 'correlation']:
            return Sampling_method(nt, nx, self.iterations)
        else:
            return Sampling_method(nt, nx)
    
    def sample(self, nt: int, nx:int) -> np.ndarray:
        '''
        Generate a Latin-hypercube design
        
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