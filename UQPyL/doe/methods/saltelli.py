import numpy as np
import warnings
from typing import Optional
from scipy.stats import qmc
from ..base import Sampler
from ._sobol import validateSobolSetup


class SaltelliDesign(Sampler):
    """
    Saltelli design for sensitivity analysis.

    Examples:
        >>> from UQPyL.problem import Sphere
        >>> problem = Sphere(nInput=3)
        >>> sampler = SaltelliDesign(secondOrder=True)
        >>> X, meta = sampler.sampleWithMeta(problem, 128, seed=11)
        >>> print(X.shape)
        (1024, 3)
        >>> print(meta["designType"], meta["secondOrder"])
        saltelli True

    References:
        [1] A. Saltelli, Making best use of model evaluations to compute sensitivity indices,
            Computer Physics Communications, 145(2):280-297, 2002,
            doi: 10.1016/S0010-4655(02)00280-1.
        [2] A. Saltelli et al., Variance based sensitivity analysis of model output. Design
            and estimator for the total sensitivity index, Computer Physics Communications,
            181(2):259-270, 2010, doi: 10.1016/j.cpc.2009.09.018.
    """

    def __init__(self, scramble: bool = True, skipValue: int = 0, secondOrder: bool = False):
        """
        Initialize the Saltelli design sampler.

        Args:
            scramble: Whether to scramble the Sobol base sequence.
            skipValue: Number of initial Sobol points to skip.
            secondOrder: Whether to generate the second-order design.
        """
        super().__init__()

        self.scramble = scramble
        self.skipValue = skipValue
        self.secondOrder = secondOrder

    def sampleWithMeta(self, problem, N: int, seed=None, *, output="real"):
        """
        Generate Saltelli samples with metadata.

        Args:
            problem: Problem instance.
            N: Base sample size.
            seed: Random seed.

        Returns:
            tuple: ``(X, meta)`` where ``X`` is the sample matrix.
        """
        self._validate_sampling_setup(N)
        return super().sampleWithMeta(problem, N, seed=seed, output=output)

    def _validate_sampling_setup(self, N: int):
        self._validate_sample_count(N)
        validateSobolSetup(N, self.skipValue)

    def _generate(self, nSamples: int, nInput: int):
        """
        Generate unit-space Saltelli samples.

        Args:
            nSamples: Base sample size.
            nInput: Number of input variables.

        Returns:
            np.ndarray: Unit-space Saltelli samples.
        """
        N = nSamples
        skipValue = self.skipValue
        calSecondOrder = self.secondOrder

        M = skipValue if skipValue > 0 else None

        sobol_seed = None
        if self.scramble:
            sobol_seed = self.rng.integers(1, 1000000)

        sampler = qmc.Sobol(nInput * 2, scramble=self.scramble, seed=sobol_seed)

        if M:
            sampler.fast_forward(M)

        if calSecondOrder:
            saltelliSequence = np.zeros(((2 * nInput + 2) * N, nInput))
        else:
            saltelliSequence = np.zeros(((nInput + 2) * N, nInput))

        # Our setup check reports quality once, independently of SciPy's
        # cumulative point count after fast-forwarding.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The balance properties of Sobol' points require n to be a power of 2\.",
                category=UserWarning,
            )
            baseSequence = sampler.random(N)

        index = 0
        for i in range(N):
            saltelliSequence[index, :] = baseSequence[i, :nInput]
            index += 1

            saltelliSequence[index : index + nInput, :] = np.tile(baseSequence[i, :nInput], (nInput, 1))
            saltelliSequence[index : index + nInput, :][np.diag_indices(nInput)] = baseSequence[i, nInput:]
            index += nInput

            if calSecondOrder:
                saltelliSequence[index : index + nInput, :] = np.tile(baseSequence[i, nInput:], (nInput, 1))
                saltelliSequence[index : index + nInput, :][np.diag_indices(nInput)] = baseSequence[i, :nInput]
                index += nInput

            saltelliSequence[index, :] = baseSequence[i, nInput : nInput * 2]
            index += 1

        xSample = saltelliSequence

        return xSample

    def _expected_shape(self, nSamples: int, nInput: int):
        """
        Return the expected Saltelli sample shape.

        Args:
            nSamples: Base sample size.
            nInput: Number of input variables.

        Returns:
            tuple: Expected sample shape.
        """
        if self.secondOrder:
            return ((2 * nInput + 2) * nSamples, nInput)
        return ((nInput + 2) * nSamples, nInput)

    def _build_meta(self, problem, nSamples: int, seed: Optional[int] = None):
        return {
            "designType": "saltelli",
            "N": nSamples,
            "secondOrder": self.secondOrder,
            "skipValue": self.skipValue,
            "scramble": self.scramble,
            "blockSize": 2 * problem.nInput + 2 if self.secondOrder else problem.nInput + 2,
            "seed": seed if self.scramble else None,
        }
