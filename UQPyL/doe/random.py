import numpy as np

from .base import Sampler

class Random(Sampler):
    """
    Uniform random sampler.
    """

    def sample(self, problem, nSamples: int, seed=None):
        return super().sample(problem, nSamples, seed=seed)

    def sampleWithMeta(self, problem, nSamples: int, seed=None):
        return super().sampleWithMeta(problem, nSamples, seed=seed)
    
    def _generate(self, nSamples: int, nInput: int):
        """
        Generate unit-space random samples.

        :param nSamples: Number of samples.
        :param nInput: Number of input variables.
        :return np.ndarray: Unit-space random samples.
        """
        H = self.rng.random((nSamples, nInput))
        
        return H

    def _build_meta(self, problem, nSamples: int, seed=None):
        return {
            "designType": "random",
            "seed": seed,
        }
