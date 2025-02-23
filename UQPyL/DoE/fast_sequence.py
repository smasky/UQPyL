import numpy as np
from typing import Union, Optional

from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

class FAST_Sequence(Sampler):
    """
    The sample technique for FAST (Fourier Amplitude Sensitivity Test) method.
    
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
            phi = 2 * np.pi * np.random.rand()
            sin_result = np.sin(w_tmp[:, None] * s + phi)
            arsin_result = (1 / np.pi) * np.arcsin(sin_result)  # Saltelli's formula
            xInit[idx, :] = 0.5 + arsin_result.transpose()
        
        return xInit
    
    @decoratorRescale
    def sample(self, nt: int, nx: Optional[int] = None, problem: Optional[Problem] = None, random_seed: Optional[int] = None):
        """
        Generate a sample for the FAST method.
        
        :param nt: Number of sample points.
        :param nx: Input dimensions of sampled points.
        :param problem: Problem instance to use bounds for sampling.
        :param random_seed: Random seed for reproducibility.
        :return: A 2D array of FAST samples.
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
    
    