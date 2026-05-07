import numpy as np
from typing import Union, Optional
from itertools import product

from ..base import Sampler
from ...problem import ProblemABC as Problem

class FFD(Sampler):
    """
    Full factorial design.

    Examples:
        >>> from UQPyL.problems import Sphere
        >>> problem = Sphere(nInput=2)
        >>> sampler = FFD()
        >>> X, meta = sampler.sampleWithMeta(problem, levels=[3, 4])
        >>> print(X.shape)
        (12, 2)
        >>> print(meta["levels"])
        [3, 4]

    References:
        [1] R. A. Fisher, The Design of Experiments, Oliver and Boyd, 1935.
        [2] G. E. P. Box, W. G. Hunter, and J. S. Hunter, Statistics for Experimenters:
            Design, Innovation, and Discovery, 2nd ed., Wiley, 2005.
    """
    
    def _generate(self, levels: Union[np.ndarray, int, list], nInput: int):
        """
        Generate the unit-space factorial grid.

        :param levels: Levels for each input variable.
        :param nInput: Number of input variables.
        :return np.ndarray: Unit-space factorial grid.
        """
        if isinstance(levels, int):
            levels = [levels] * nInput
        elif isinstance(levels, np.ndarray):
            levels = levels.ravel().tolist()
        
        if len(levels) != nInput:
            raise ValueError("The length of levels must match nInput or be a scalar.")
        
        factor_levels = [np.linspace(0, 1, num=level)[:level] for level in levels]
        factor_combinations = list(product(*factor_levels))
       
        H = np.array(factor_combinations)
        
        return H
    
    def sample(self, problem: Problem, levels: Union[np.ndarray, int, list], seed: Optional[int] = None):
        """
        Generate a full factorial sample.

        :param problem: Problem instance.
        :param levels: Levels for each input variable.
        :param seed: Random seed kept for API consistency. It does not affect the result.
        :return np.ndarray: Samples in the problem space.
        """
        X, _ = self.sampleWithMeta(problem, levels, seed=seed)
        return X

    def sampleWithMeta(self, problem: Problem, levels: Union[np.ndarray, int, list], seed: Optional[int] = None):
        """
        Generate a full factorial sample set with metadata.

        :param problem: Problem instance.
        :param levels: Levels for each input variable.
        :param seed: Random seed kept for API consistency. It does not affect the result.
        :return tuple: ``(X, meta)`` where ``X`` is the sample matrix.
        """

        self._validate_problem(problem)
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        nInput = problem.nInput
        U = self._generate(levels, nInput)
        U = self._validate_generated_samples(U, (U.shape[0], nInput))
        X = problem.unit_to_space(U)

        meta = self._build_meta(problem, levels, seed=seed)
        return X, meta

    def _build_meta(self, problem: Problem, levels: Union[np.ndarray, int, list], seed: Optional[int] = None):
        if isinstance(levels, np.ndarray):
            levels = levels.ravel().tolist()
        elif isinstance(levels, int):
            levels = [levels] * problem.nInput

        return {
            "designType": "full_fact",
            "levels": levels,
            "seed": None,
        }
