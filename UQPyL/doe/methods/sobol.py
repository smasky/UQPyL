import warnings
from scipy.stats.qmc import Sobol as QmcSobol

from ..base import Sampler
from ._sobol import validateSobolSetup


class Sobol(Sampler):
    """
    Sobol low-discrepancy sampler.

    Examples:
        >>> from UQPyL.problem import Sphere
        >>> problem = Sphere(nInput=4)
        >>> sampler = Sobol(scramble=True, skipValue=16)
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

        Args:
            scramble: Whether to scramble the Sobol sequence.
            skipValue: Number of initial Sobol points to skip.
        """

        super().__init__()

        self.scramble = scramble

        self.skipValue = skipValue

    def sampleWithMeta(self, problem, nSamples: int, seed=None, *, output="real"):
        """
        Generate Sobol samples with metadata.

        Args:
            problem: Problem instance.
            nSamples: Number of samples.
            seed: Random seed.

        Returns:
            tuple: ``(X, meta)`` where ``X`` is the sample matrix.
        """
        self._validate_sampling_setup(nSamples)
        return super().sampleWithMeta(problem, nSamples, seed=seed, output=output)

    def _validate_sampling_setup(self, nSamples: int):
        self._validate_sample_count(nSamples)
        validateSobolSetup(nSamples, self.skipValue)

    def _generate(self, nSamples: int, nInput: int):
        """
        Generate unit-space Sobol samples.

        Args:
            nSamples: Number of samples.
            nInput: Number of input variables.

        Returns:
            np.ndarray: Unit-space Sobol samples.
        """
        sobol_seed = None
        if self.scramble:
            sobol_seed = self.rng.integers(1, 1000000)

        sampler = QmcSobol(d=nInput, scramble=self.scramble, seed=sobol_seed)

        if self.skipValue:
            sampler.fast_forward(self.skipValue)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The balance properties of Sobol' points require n to be a power of 2\.",
                category=UserWarning,
            )
            xInit = sampler.random(nSamples)

        return xInit

    def _build_meta(self, problem, nSamples: int, seed=None):
        return {
            "designType": "sobol_sequence",
            "scramble": self.scramble,
            "skipValue": self.skipValue,
            "seed": seed if self.scramble else None,
        }
