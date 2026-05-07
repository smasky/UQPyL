import numpy as np
import warnings
from typing import Optional

from ..base import Sampler

class FASTDesign(Sampler):
    """
    FAST design for sensitivity analysis.

    Examples:
        >>> from UQPyL.problems import Sphere
        >>> problem = Sphere(nInput=3)
        >>> sampler = FASTDesign(M=4)
        >>> X, meta = sampler.sampleWithMeta(problem, 128, seed=5)
        >>> print(X.shape)
        (384, 3)
        >>> print(meta["M"])
        4

    References:
        [1] R. I. Cukier, C. M. Fortuin, K. E. Shuler, A. G. Petschek, and J. H. Schaibly,
            Study of the sensitivity of coupled reaction systems to uncertainties in rate
            coefficients. I Theory, The Journal of Chemical Physics, 59(8):3873-3878, 1973,
            doi: 10.1063/1.1680571.
        [2] A. Saltelli, S. Tarantola, and K. P.-S. Chan, A Quantitative Model-Independent
            Method for Global Sensitivity Analysis of Model Output, Technometrics,
            41(1):39-56, 1999, doi: 10.1080/00401706.1999.10485594.
    """
    
    def __init__(self, M: int = 4):
        """
        Initialize the FAST design sampler.

        :param M: Interference parameter.
        """
        super().__init__()
        
        self.M = M

    def sampleWithMeta(self, problem, N: int, seed=None):
        """
        Generate FAST samples with metadata.

        :param problem: Problem instance.
        :param N: Base sample size.
        :param seed: Random seed.
        :return tuple: ``(X, meta)`` where ``X`` is the sample matrix.
        """
        self._validate_sampling_setup(N)
        return super().sampleWithMeta(problem, N, seed=seed)

    def _validate_sampling_setup(self, N: int):
        if not isinstance(self.M, int):
            raise TypeError("M must be an integer.")

        if self.M <= 0:
            raise ValueError(f"M must be greater than 0. Received M={self.M}.")

        if N <= 4 * self.M**2:
            raise ValueError(
                f"FAST requires N > 4*M^2. Received N={N}, M={self.M}."
            )
    
    def _generate(self, nSamples: int, nInput: int):
        """
        Generate unit-space FAST samples.

        :param nSamples: Base sample size.
        :param nInput: Number of input variables.
        :return np.ndarray: Unit-space FAST samples.
        """
        
        w = np.zeros(nInput)
        w[0] = np.floor((nSamples - 1) / (2 * self.M))
        max_wi = np.floor(w[0] / (2 * self.M))  # Saltelli's method

        if max_wi < 1:
            raise ValueError(
                f"N={nSamples} is too small for M={self.M}; the auxiliary FAST frequencies become invalid."
            )

        if max_wi < nInput - 1:
            warnings.warn(
                f"FASTDesign will reuse auxiliary frequencies because N={nSamples}, M={self.M}, "
                f"and nInput={nInput} do not provide enough unique frequencies. "
                "This configuration is still allowed, but spectral aliasing risk increases.",
                UserWarning,
                stacklevel=2,
            )
        
        if max_wi >= nInput - 1:
            w[1:] = np.floor(np.linspace(1, max_wi, nInput - 1))
        else:
            w[1:] = np.arange(nInput - 1) % max_wi + 1
        
        s = (2 * np.pi / nSamples) * np.arange(nSamples)
        
        xInit = np.zeros((nSamples * nInput, nInput))
        w_tmp = np.zeros(nInput)
        
        for i in range(nInput):
            w_tmp[i] = w[0]
            idx = list(range(i)) + list(range(i + 1, nInput))
            w_tmp[idx] = w[1:]
            idx = range(i * nSamples, (i + 1) * nSamples)
            phi = 2 * np.pi * self.rng.random()
            sin_result = np.sin(w_tmp[:, None] * s + phi)
            arsin_result = (1 / np.pi) * np.arcsin(sin_result)  # Saltelli's formula
            xInit[idx, :] = 0.5 + arsin_result.transpose()
        
        return xInit

    def _expected_shape(self, nSamples: int, nInput: int):
        """
        Return the expected FAST sample shape.

        :param nSamples: Base sample size.
        :param nInput: Number of input variables.
        :return tuple: Expected sample shape.
        """
        return (nSamples * nInput, nInput)

    def _build_meta(self, problem, nSamples: int, seed: Optional[int] = None):
        return {
            "designType": "fast",
            "N": nSamples,
            "M": self.M,
            "blockSize": nSamples,
            "seed": seed,
        }
    
