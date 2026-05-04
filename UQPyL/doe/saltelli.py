import numpy as np
import warnings
from typing import Optional
from scipy.stats import qmc
from .base import Sampler

class SaltelliDesign(Sampler):
    """
    Saltelli design for sensitivity analysis.
    """
    
    def __init__(self, scramble: bool = True, skipValue: int = 0, secondOrder: bool = False):
        """
        Initialize the Saltelli design sampler.

        :param scramble: Whether to scramble the Sobol base sequence.
        :param skipValue: Number of initial Sobol points to skip.
        :param secondOrder: Whether to generate the second-order design.
        """
        super().__init__()
        
        self.scramble = scramble
        self.skipValue = skipValue
        self.secondOrder = secondOrder

    def sampleWithMeta(self, problem, N: int, seed=None):
        """
        Generate Saltelli samples with metadata.

        :param problem: Problem instance.
        :param N: Base sample size.
        :param seed: Random seed.
        :return tuple: ``(X, meta)`` where ``X`` is the sample matrix.
        """
        self._validate_sampling_setup(N)
        return super().sampleWithMeta(problem, N, seed=seed)

    def _validate_sampling_setup(self, N: int):
        if not isinstance(self.skipValue, int):
            raise TypeError("skipValue must be an integer.")

        if self.skipValue < 0:
            raise ValueError("skipValue must be greater than or equal to 0.")

        if N < self.skipValue:
            raise ValueError(
                f"N must be greater than or equal to skipValue. "
                f"Received N={N}, skipValue={self.skipValue}."
            )

        if N > 0 and (N & (N - 1)) != 0:
            next_power = int(np.power(2, np.ceil(np.log2(N))))
            warnings.warn(
                f"Saltelli designs are usually built from a Sobol base size that is a power of 2. "
                f"Received N={N}; consider using {next_power}.",
                UserWarning,
                stacklevel=2,
            )

        if self.skipValue > 0 and (self.skipValue & (self.skipValue - 1)) != 0:
            warnings.warn(
                "Saltelli designs usually use a power-of-2 skipValue for better Sobol balance. "
                f"Received skipValue={self.skipValue}.",
                UserWarning,
                stacklevel=2,
            )
    
    def _generate(self, nSamples: int, nInput: int):
        """
        Generate unit-space Saltelli samples.

        :param nSamples: Base sample size.
        :param nInput: Number of input variables.
        :return np.ndarray: Unit-space Saltelli samples.
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
        
        baseSequence = sampler.random(N)
        
        index = 0
        for i in range(N):
            saltelliSequence[index, :] = baseSequence[i, :nInput]
            index += 1
            
            saltelliSequence[index:index + nInput, :] = np.tile(baseSequence[i, :nInput], (nInput, 1))
            saltelliSequence[index:index + nInput, :][np.diag_indices(nInput)] = baseSequence[i, nInput:]               
            index += nInput
           
            if calSecondOrder:
                saltelliSequence[index:index + nInput, :] = np.tile(baseSequence[i, nInput:], (nInput, 1))
                saltelliSequence[index:index + nInput, :][np.diag_indices(nInput)] = baseSequence[i, :nInput] 
                index += nInput
            
            saltelliSequence[index, :] = baseSequence[i, nInput:nInput * 2]
            index += 1
        
        xSample = saltelliSequence
        
        return xSample

    def _expected_shape(self, nSamples: int, nInput: int):
        """
        Return the expected Saltelli sample shape.

        :param nSamples: Base sample size.
        :param nInput: Number of input variables.
        :return tuple: Expected sample shape.
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
