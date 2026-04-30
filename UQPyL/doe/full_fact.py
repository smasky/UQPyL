import numpy as np
from typing import Union, Optional
from itertools import product

from .base import Sampler
from ..problem import ProblemABC as Problem

class FFD(Sampler):
    """
    Full factorial design on the unit hypercube.

    Unlike the other general DOE samplers, this sampler is driven by
    per-dimension ``levels`` instead of a target sample count ``nt``.
    """
    
    def _generate(self, levels: Union[np.ndarray, int, list], nx: int):
        """
        Internal method to generate the full factorial design.
        
        :param levels: Levels for each input dimension. Can be an integer, list, or ndarray.
        :param nx: Number of input dimensions.
        :return: A 2D array of full factorial design samples.
        """
        if isinstance(levels, int):
            levels = [levels] * nx
        elif isinstance(levels, np.ndarray):
            levels = levels.ravel().tolist()
        
        if len(levels) != nx:
            raise ValueError('The length of levels should be equal to nx or 1')
        
        factor_levels = [np.linspace(0, 1, num=level)[:level] for level in levels]
        factor_combinations = list(product(*factor_levels))
       
        H = np.array(factor_combinations)
        
        return H
    
    def sample(self, problem: Problem, levels: Union[np.ndarray, int, list], seed: Optional[int] = None):
        """
        Generate a full factorial design sample.
        
        :param problem: Problem instance to use bounds for sampling.
        :param levels: Levels for each input dimension. Can be an integer, list, or ndarray.
        :param seed: Optional random seed kept for API consistency. It does not affect the result.
        
        :return: A 2D array of full factorial design samples.
        """

        self._validate_problem(problem)
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        nx = problem.nInput
        X = self._generate(levels, nx)
        X = self._validate_generated_samples(X, X.shape[0], nx)

        return problem.unit_to_space(X)
