import numpy as np
from typing import Union
from itertools import product

from .sampler_ABC import Sampler

class FFD(Sampler):
    '''
    Full Factorial Design
    
    Methods:
    __call__ or sample: Generate a Latin-hypercube design
    
    Examples:
        >>> ffd=FFD()
        >>> samples=ffd(3, [2,3,4]) or samples=ffd.sample(3, [2,3,4])
    '''
    
    def __call__(self, nx: int, levels: Union[np.ndarray, int, list]) -> np.ndarray:
        
        return self._generate(nx, levels)
    
    def _generate(self, nx: int, levels: Union[np.ndarray, int, list]) -> np.ndarray:

        if isinstance(levels, int):
            levels = [levels]*nx
        elif isinstance(levels, np.ndarray):
            levels = levels.ravel().tolist()
        
        if len(levels)!=nx:
            raise ValueError('The length of levels should be equal to nx or 1')
        
        factor_levels = [np.linspace(0, 1, num=level + 1)[:level] for level in levels]
               
        factor_combinations = list(product(*factor_levels))
       
        H = np.array(factor_combinations)
        
        return H
    
    def sample(self, nx: int, levels: Union[np.ndarray, int, list]) -> np.ndarray:
        '''
        Parameters:
        nx: int
            The number of input dimensions
        
        levels: Union[np.ndarray, int, list]
            The levels for each input dimension
            
        Returns:
        H: 2d-array
            An n-by-samples design matrix between zero and one.        
        '''
        
        return self._generate(nx, levels)
        
        