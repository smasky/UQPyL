import numpy as np
from scipy.stats.qmc import Sobol as QmcSobol

from .base import Sampler

class Sobol(Sampler):
    """
    Sobol low-discrepancy sampler.

    The public ``sample()`` method is inherited from :class:`Sampler`.
    """
    
    def __init__(self, scramble: bool = True, skipValue: int = 0):
        """
        Initialize the Sobol sampler.
        
        :param scramble: Whether to scramble the Sobol sequence.
        :param skipValue: Number of initial points to skip in the sequence.
        """
        
        super().__init__()
        
        self.scramble = scramble
        
        self.skipValue = skipValue
        
    def _generate(self, nt: int, nx: int):
        """
        Generate unit-space Sobol samples.
        
        :param nt: Number of sampled points.
        :param nx: Input dimensions of sampled points.
        :return: A 2D array of shape ``(nt, nx)`` in the unit hypercube.
        """
        sobol_seed = self.rng.integers(1, 1000000)
        
        sampler = QmcSobol(d=nx, scramble=self.scramble, seed=sobol_seed)
        
        xInit = sampler.random(nt + self.skipValue)
        
        return xInit[self.skipValue:, :]


# Compatibility alias retained during DOE API cleanup.
SobolSequence = Sobol
