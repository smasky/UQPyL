import numpy as np
from typing import Union, Optional
from itertools import product

from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

class FFD(Sampler):
    '''
    Full Factorial Design
    
    Methods:
    __call__ or sample: Generate a Latin-hypercube design

    '''
    
    #Abandon
    # def __call__(self, nx: int, levels: Union[np.ndarray, int, list]) -> np.ndarray:
        
    #     return self._generate(nx, levels)
    
    def _generate(self, levels: Union[np.ndarray, int, list], nx: int):

        if isinstance(levels, int):
            levels = [levels]*nx
        elif isinstance(levels, np.ndarray):
            levels = levels.ravel().tolist()
        
        if len(levels)!=nx:
            raise ValueError('The length of levels should be equal to nx or 1')
        
        factor_levels = [np.linspace(0, 1, num=level)[:level] for level in levels]
               
        factor_combinations = list(product(*factor_levels))
       
        H = np.array(factor_combinations)
        
        return H
    
    @decoratorRescale
    def sample(self, levels: Union[np.ndarray, int, list], nx: Optional[int] = None, problem: Optional[Problem] = None, random_seed: Optional[int] = None):
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
        
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
        
        if problem is not None and nx is not None:
            if(problem.nInput!=nx):
                raise ValueError('The input dimensions of the problem and the samples must be the same')
        elif problem is None and nx is None:
            raise ValueError('Either the problem or the input dimensions must be provided')
        
        nx=problem.nInput if problem is not None else nx

        return self._generate(levels, nx)