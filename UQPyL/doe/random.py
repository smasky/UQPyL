import numpy as np

from .base import Sampler

class Random(Sampler):
    """
    Uniform random sampler in the unit hypercube.

    The public ``sample()`` method is inherited from :class:`Sampler`.
    """
    
    def _generate(self, nt: int, nx: int):
        """
        Generate unit-space random samples.
        
        :param nt: Number of sampled points.
        :param nx: Input dimensions of sampled points.
        :return: A 2D array of shape ``(nt, nx)`` with values in ``[0, 1)``.
        """
        H = self.rng.random((nt, nx))
        
        return H
