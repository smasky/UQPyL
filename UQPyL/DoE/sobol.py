import numpy as np
from scipy.stats.qmc import Sobol
from typing import Optional

from .base import Sampler
from ..problem import ProblemABC as Problem

class SobolSequence(Sampler):
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
    
    # @decoratorRescale
    def sample(self, problem: Problem, nt: int, random_seed: Optional[int] = None):
        """
        Generate a Sobol sequence sample.
        
        :param problem: Problem instance to use bounds for sampling.
        :param nt: Number of sampled points.
        :param random_seed: Random seed for reproducibility.
        
        :return: A 2D array of Sobol sequence samples.
        """
                
        self.random_state = np.random.RandomState(random_seed) if random_seed is not None else np.random.RandomState()
        
        nx = problem.nInput
        
        return self._generate(nt, nx)   