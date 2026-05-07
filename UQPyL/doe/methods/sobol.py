import numpy as np
import warnings
from scipy.stats.qmc import Sobol as QmcSobol

from ..base import Sampler

class Sobol(Sampler):
    """
    Sobol low-discrepancy sampler.

    Examples:
        >>> from UQPyL.problems import Sphere
        >>> problem = Sphere(nInput=4)
        >>> sampler = Sobol(scramble=True, skipValue=8)
        >>> X, meta = sampler.sampleWithMeta(problem, 16, seed=7)
        >>> print(X.shape)
        (16, 4)
        >>> print(meta["designType"])
        sobol_sequence

    References:
        [1] I. M. Sobol', The distribution of points in a cube and the accurate evaluation
            of integrals, USSR Computational Mathematics and Mathematical Physics,
            7(4):86-112, 1967, doi: 10.1016/0041-5553(67)90144-9.
        [2] SciPy QMC Sobol engine documentation,
            https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
    """
    
    def __init__(self, scramble: bool = True, skipValue: int = 0):
        """
        Initialize the Sobol sampler.

        :param scramble: Whether to scramble the Sobol sequence.
        :param skipValue: Number of initial Sobol points to skip.
        """
        
        super().__init__()
        
        self.scramble = scramble
        
        self.skipValue = skipValue

    def sampleWithMeta(self, problem, nSamples: int, seed=None):
        """
        Generate Sobol samples with metadata.

        :param problem: Problem instance.
        :param nSamples: Number of samples.
        :param seed: Random seed.
        :return tuple: ``(X, meta)`` where ``X`` is the sample matrix.
        """
        self._validate_sampling_setup(nSamples)
        return super().sampleWithMeta(problem, nSamples, seed=seed)

    def _validate_sampling_setup(self, nSamples: int):
        if not isinstance(self.skipValue, int):
            raise TypeError("skipValue must be an integer.")

        if self.skipValue < 0:
            raise ValueError("skipValue must be greater than or equal to 0.")

        if nSamples < self.skipValue:
            raise ValueError(
                f"nSamples must be greater than or equal to skipValue. "
                f"Received nSamples={nSamples}, skipValue={self.skipValue}."
            )

        if nSamples > 0 and (nSamples & (nSamples - 1)) != 0:
            next_power = int(np.power(2, np.ceil(np.log2(nSamples))))
            warnings.warn(
                f"Sobol sequences are best balanced when nSamples is a power of 2. "
                f"Received nSamples={nSamples}; consider using {next_power}.",
                UserWarning,
                stacklevel=2,
            )

        if self.skipValue > 0 and (self.skipValue & (self.skipValue - 1)) != 0:
            warnings.warn(
                "Sobol sequences usually use a power-of-2 skipValue for better balance. "
                f"Received skipValue={self.skipValue}.",
                UserWarning,
                stacklevel=2,
            )
        
    def _generate(self, nSamples: int, nInput: int):
        """
        Generate unit-space Sobol samples.

        :param nSamples: Number of samples.
        :param nInput: Number of input variables.
        :return np.ndarray: Unit-space Sobol samples.
        """
        sobol_seed = None
        if self.scramble:
            sobol_seed = self.rng.integers(1, 1000000)
        
        sampler = QmcSobol(d=nInput, scramble=self.scramble, seed=sobol_seed)
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The balance properties of Sobol' points require n to be a power of 2\.",
                category=UserWarning,
            )
            xInit = sampler.random(nSamples + self.skipValue)
        
        return xInit[self.skipValue:, :]

    def _build_meta(self, problem, nSamples: int, seed=None):
        return {
            "designType": "sobol_sequence",
            "scramble": self.scramble,
            "skipValue": self.skipValue,
            "seed": seed if self.scramble else None,
        }

