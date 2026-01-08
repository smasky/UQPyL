import numpy as np
from typing import Optional

from .base import Sampler
from ..problem import ProblemABC as Problem

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
        H = self.rng.random((nt, nx))
        
        return H
    
    def sample(self, problem: Problem, nt: int, seed: Optional[int] = None):
        """
        Generate a sample with random values between zero and one.
        
        :param problem: Problem instance to use bounds for sampling.
        :param nt: Number of sampled points.
        :param random_seed: Random seed for reproducibility.
        :return: A 2D array of random samples.
        """
        
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        nx = problem.nInput
        
        return problem._transform_unit_X(self._generate(nt, nx))