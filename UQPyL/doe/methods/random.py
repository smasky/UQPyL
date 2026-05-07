import numpy as np

from ..base import Sampler

class Random(Sampler):
    """
    Uniform random sampler.

    Examples:
        >>> from UQPyL.problems import Sphere
        >>> problem = Sphere(nInput=2)
        >>> sampler = Random()
        >>> X, meta = sampler.sampleWithMeta(problem, 5, seed=42)
        >>> print(X.shape)
        (5, 2)
        >>> print(meta["designType"])
        random

    References:
        [1] G. E. P. Box, W. G. Hunter, and J. S. Hunter, Statistics for Experimenters:
            Design, Innovation, and Discovery, 2nd ed., Wiley, 2005.
    """

    def sample(self, problem, nSamples: int = None, seed=None, nt: int = None):
        return super().sample(problem, nSamples, seed=seed, nt=nt)

    def sampleWithMeta(self, problem, nSamples: int = None, seed=None, nt: int = None):
        return super().sampleWithMeta(problem, nSamples, seed=seed, nt=nt)
    
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
