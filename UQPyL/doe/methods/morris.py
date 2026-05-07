import numpy as np
from typing import Optional

from ..base import Sampler

class MorrisDesign(Sampler):
    """
    Morris design for sensitivity analysis.

    Examples:
        >>> from UQPyL.problems import Sphere
        >>> problem = Sphere(nInput=3)
        >>> sampler = MorrisDesign(numLevels=4)
        >>> X, meta = sampler.sampleWithMeta(problem, 10, seed=21)
        >>> print(X.shape)
        (40, 3)
        >>> print(meta["trajectorySize"])
        4

    References:
        [1] Max D. Morris, Factorial Sampling Plans for Preliminary Computational
            Experiments, Technometrics, 33(2):161-174, 1991,
            doi: 10.2307/1269043.
        [2] A. Saltelli, K. Chan, and E. M. Scott, eds., Sensitivity Analysis,
            Wiley, 2000.
    """
    
    def __init__(self, numLevels: int = 4):
        """
        Initialize the Morris design sampler.

        :param numLevels: Number of Morris levels.
        """
        super().__init__()
        
        self.numLevels = numLevels

    def sampleWithMeta(self, problem, numTrajectory: int, seed=None):
        """
        Generate Morris samples with metadata.

        :param problem: Problem instance.
        :param numTrajectory: Number of trajectories.
        :param seed: Random seed.
        :return tuple: ``(X, meta)`` where ``X`` is the sample matrix.
        """
        self._validate_num_levels()
        return super().sampleWithMeta(problem, numTrajectory, seed=seed)

    def _validate_num_levels(self):
        if not isinstance(self.numLevels, int):
            raise TypeError("numLevels must be an integer.")

        if self.numLevels < 4:
            raise ValueError(
                f"numLevels must be greater than or equal to 4. Received numLevels={self.numLevels}."
            )

        if self.numLevels % 2 != 0:
            raise ValueError(
                f"numLevels must be an even integer. Received numLevels={self.numLevels}."
            )
        
    def _generate(self, nSamples: int, nInput: int):
        """
        Generate unit-space Morris samples.

        :param nSamples: Number of trajectories.
        :param nInput: Number of input variables.
        :return np.ndarray: Unit-space Morris samples.
        """
        xInit = np.zeros((nSamples * (nInput + 1), nInput))
        
        for i in range(nSamples):
            xInit[i * (nInput + 1):(i + 1) * (nInput + 1), :] = self._generate_trajectory(nInput)
        
        return xInit

    def _expected_shape(self, nSamples: int, nInput: int):
        """
        Return the expected Morris sample shape.

        :param nSamples: Number of trajectories.
        :param nInput: Number of input variables.
        :return tuple: Expected sample shape.
        """
        return (nSamples * (nInput + 1), nInput)
        
    def _generate_trajectory(self, nInput: int):
        """
        Generate a single Morris trajectory.

        :param nInput: Number of input variables.
        :return np.ndarray: One unit-space trajectory.
        """
        delta = self.numLevels / (2 * (self.numLevels - 1))
        
        B = np.tril(np.ones([nInput + 1, nInput], dtype=int), -1)
        
        # From paper[1] page 164
        D_star = np.diag(self.rng.choice([-1, 1], nInput))  # Step 1
        J = np.ones((nInput + 1, nInput))
        
        levels_grids = np.linspace(0, 1 - delta, int(self.numLevels / 2))
        x_star = self.rng.choice(levels_grids, nInput).reshape(1, -1)  # Step 2
        
        P_star = np.zeros((nInput, nInput))
        cols = self.rng.choice(nInput, nInput, replace=False)
        P_star[np.arange(nInput), cols] = 1  # Step 3
        
        element_a = J[0, :] * x_star
        element_b = P_star.T
        element_c = np.matmul(2.0 * B, element_b)
        element_d = np.matmul((element_c - J), D_star)

        B_star = element_a + (delta / 2.0) * (element_d + J)
    
        return B_star

    def _build_meta(self, problem, nSamples: int, seed: Optional[int] = None):
        return {
            "designType": "morris",
            "numTrajectory": nSamples,
            "numLevels": self.numLevels,
            "trajectorySize": problem.nInput + 1,
            "seed": seed,
        }
        
        
        
        
    
