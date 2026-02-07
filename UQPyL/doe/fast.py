import numpy as np
from typing import Union, Optional

from .base import Sampler
from ..problem import ProblemABC as Problem

class FASTSequence(Sampler):
    """
    The sample method for FAST (Fourier Amplitude Sensitivity Test) method.
    
    This class generates samples for the FAST method, which is used for sensitivity analysis
    by decomposing the output variance into contributions from each input variable.
    
    Methods:
        sample: Generate a sample for the FAST method.
    """
    
    def __init__(self, M: int = 4):
        """
        Initialize the FAST Sequence sampler with a specified interference parameter.
        
        :param M: The interference parameter for the Fourier series decomposition.
        """
        super().__init__()
        
        self.M = M
    
    def _generate(self, nt: int, nx: int):
        """
        Generate a sample for the FAST method.
        
        :param nt: Number of sample points.
        :param nx: Input dimensions of sampled points.
        
        :return: A 2D array of samples, normalized so factor values are uniformly spaced between zero and one.
        """
        
        if nt <= 4 * self.M**2:
            raise ValueError("The number of samples must be greater than 4 * M^2!")
        
        w = np.zeros(nx)
        w[0] = np.floor((nt - 1) / (2 * self.M))
        max_wi = np.floor(w[0] / (2 * self.M))  # Saltelli's method
        
        if max_wi >= nx - 1:
            w[1:] = np.floor(np.linspace(1, max_wi, nx - 1))
        else:
            w[1:] = np.arange(nx - 1) % max_wi + 1
        
        s = (2 * np.pi / nt) * np.arange(nt)
        
        xInit = np.zeros((nt * nx, nx))
        w_tmp = np.zeros(nx)
        
        for i in range(nx):
            w_tmp[i] = w[0]
            idx = list(range(i)) + list(range(i + 1, nx))
            w_tmp[idx] = w[1:]
            idx = range(i * nt, (i + 1) * nt)
            phi = 2 * np.pi * self.rng.random()
            sin_result = np.sin(w_tmp[:, None] * s + phi)
            arsin_result = (1 / np.pi) * np.arcsin(sin_result)  # Saltelli's formula
            xInit[idx, :] = 0.5 + arsin_result.transpose()
        
        return xInit
    
    def sample(self, problem: Problem, nt: int, seed: int = None):
        """
        Generate a sample for the FAST method.
        
        :param problem: Problem instance to use bounds for sampling.
        :param nt: Number of sample points.
        :param nx: Input dimensions of sampled points.
        :param random_seed: Random seed for reproducibility.
        
        :return: A 2D array of FAST samples.
        """
        
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        nx = problem.nInput
        
        return problem._transform_unit_X(self._generate(nt, nx))
    
    