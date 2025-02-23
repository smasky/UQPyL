import numpy as np
from typing import Optional

from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

class Morris_Sequence(Sampler):
    """
    The sample technique for Morris analysis.
    
    Methods:
        sample: Generate a sample for the Morris method.
    
    Examples:
        >>> mor_seq = Morris_Sequence(numLevels=4)
        >>> mor_seq.sample(100, 4) or mor_seq(100, 4)
    
    Reference:
        [1] Max D. Morris (1991) Factorial Sampling Plans for Preliminary Computational Experiments, Technometrics, 33:2, 161-174
    """
    def __init__(self, numLevels: int = 4):
        """
        Initialize the Morris Sequence sampler with a specified number of levels.
        
        :param numLevels: Number of levels for the Morris method.
        """
        super().__init__()
        self.numLevels = numLevels
        
    def _generate(self, nt: int, nx: int):
        """
        Generate a sample for the Morris method.
        
        :param nt: Number of trajectories.
        :param nx: Input dimensions of sampled points.
        :return: A 2D array of samples, normalized so factor values are uniformly spaced between zero and one.
        """
        xInit = np.zeros((nt * (nx + 1), nx))
        
        for i in range(nt):
            xInit[i * (nx + 1):(i + 1) * (nx + 1), :] = self._generate_trajectory(nx)
        
        return xInit
    
    @decoratorRescale
    def sample(self, nt: int, nx: Optional[int] = None, problem: Optional[Problem] = None, random_seed: Optional[int] = None):
        """
        Generate a sample for the Morris method.
        
        :param nt: Number of trajectories.
        :param nx: Input dimensions of sampled points.
        :param problem: Problem instance to use bounds for sampling.
        :param random_seed: Random seed for reproducibility.
        :return: A 2D array of samples.
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
        
    def _generate_trajectory(self, nx: int):
        """
        Generate a single trajectory for the Morris method.
        
        :param nx: Input dimensions of sampled points.
        :return: A 2D array representing a single trajectory.
        """
        delta = self.numLevels / (2 * (self.numLevels - 1))
        
        B = np.tril(np.ones([nx + 1, nx], dtype=int), -1)
        
        # From paper[1] page 164
        D_star = np.diag(np.random.choice([-1, 1], nx))  # Step 1
        J = np.ones((nx + 1, nx))
        
        levels_grids = np.linspace(0, 1 - delta, int(self.numLevels / 2))
        x_star = np.random.choice(levels_grids, nx).reshape(1, -1)  # Step 2
        
        P_star = np.zeros((nx, nx))
        cols = np.random.choice(nx, nx, replace=False)
        P_star[np.arange(nx), cols] = 1  # Step 3
        
        element_a = J[0, :] * x_star
        element_b = P_star.T
        element_c = np.matmul(2.0 * B, element_b)
        element_d = np.matmul((element_c - J), D_star)

        B_star = element_a + (delta / 2.0) * (element_d + J)
    
        return B_star
        
        
        
        
    