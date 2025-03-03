import numpy as np
from typing import Optional

from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

class Random(Sampler):
    """
    Random Design
    
    Methods:
        sample: Generate a random design.
    
    Examples:
        >>> random = Random()
        >>> random.sample(10, 10) or random(10, 10)
    """
    def _generate(self, nt: int, nx: int):
        """
        Generate a random sample.
        
        :param nt: Number of sampled points.
        :param nx: Input dimensions of sampled points.
        :return: A 2D array of random samples.
        """
        H = np.random.random((nt, nx))
        
        return H
    
    @decoratorRescale
    def sample(self, nt: int, nx: Optional[int] = None, problem: Optional[Problem] = None, random_seed: Optional[int] = None):
        """
        Generate a sample with random values between zero and one.
        
        :param nt: Number of sampled points.
        :param nx: Input dimensions of sampled points.
        :param problem: Problem instance to use bounds for sampling.
        :param random_seed: Random seed for reproducibility.
        :return: A 2D array of random samples.
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