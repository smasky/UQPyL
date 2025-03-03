import numpy as np
from scipy.stats.qmc import Sobol
from typing import Optional

from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

class Sobol_Sequence(Sampler):
    """
    Sobol Sequence for quasi-random sampling.
    
    This class generates samples using the Sobol sequence, which is a low-discrepancy sequence
    used for quasi-random sampling in high-dimensional spaces.
    
    Methods:
        sample: Generate a Sobol sequence sample.
    """
    
    def __init__(self, scramble: bool = True, skipValue: int = 0):
        """
        Initialize the Sobol Sequence sampler.
        
        :param scramble: Whether to scramble the Sobol sequence.
        :param skipValue: Number of initial points to skip in the sequence.
        """
        super().__init__()
        
        self.scramble = scramble
        self.skipValue = skipValue
        
    def _generate(self, nt: int, nx: int):
        """
        Internal method to generate the Sobol sequence.
        
        :param nt: Number of sampled points.
        :param nx: Input dimensions of sampled points.
        :return: A 2D array of Sobol sequence samples.
        """
        sampler = Sobol(d=nx, scramble=self.scramble)
        xInit = sampler.random(nt + self.skipValue)
        
        return xInit[self.skipValue:, :]
    
    @decoratorRescale
    def sample(self, nt: int, nx: Optional[int] = None, problem: Optional[Problem] = None, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a Sobol sequence sample.
        
        :param nt: Number of sampled points.
        :param nx: Input dimensions of sampled points.
        :param problem: Problem instance to use bounds for sampling.
        :param random_seed: Random seed for reproducibility.
        :return: A 2D array of Sobol sequence samples.
        """
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
        
        if problem is not None and nx is not None:
            if problem.nInput != nx:
                raise ValueError('The input dimensions of the problem and the samples must be the same')
        elif problem is None and nx is None:
            raise ValueError('Either the problem or the input dimensions must be provided')
        
        nx = problem.nInput if problem is not None else nx
        
        return self._generate(nt, nx)   