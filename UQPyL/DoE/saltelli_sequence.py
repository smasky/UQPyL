import numpy as np
from typing import Optional
from scipy.stats import qmc
from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

class Saltelli_Sequence(Sampler):
    """
    Saltelli Sequence for sensitivity analysis.
    
    This class generates samples using the Saltelli method, which is an extension of the Sobol sequence
    for sensitivity analysis, allowing for the calculation of first and second order effects.
    
    Methods:
        sample: Generate a Saltelli sequence sample.
    """
    
    def __init__(self, scramble: bool = True, skipValue: int = 0, calSecondOrder: bool = False):
        """
        Initialize the Saltelli Sequence sampler.
        
        :param scramble: Whether to scramble the Sobol sequence.
        :param skipValue: Number of initial points to skip in the sequence.
        :param calSecondOrder: Whether to calculate second order effects.
        """
        super().__init__()
        
        self.scramble = scramble
        self.skipValue = skipValue
        self.calSecondOrder = calSecondOrder
        
    @decoratorRescale
    def sample(self, nt: int, nx: Optional[int] = None, problem: Optional[Problem] = None, random_seed: Optional[int] = None):
        """
        Generate a Saltelli sequence sample.
        
        :param nt: Number of base samples.
        :param nx: Input dimensions of sampled points.
        :param problem: Problem instance to use bounds for sampling.
        :param random_seed: Random seed for reproducibility.
        :return: A 2D array of Saltelli sequence samples.
        """
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()

        if problem is not None and nx is not None:
            if problem.nInput != nx:
                raise ValueError('The input dimensions of the problem and the samples must be the same')
        elif problem is None and nx is None:
            raise ValueError('Either the problem or the input dimensions must be provided')
        
        nx = problem.nInput if problem is not None else nx
        
        return self._generate(nt, nx)
    
    def _generate(self, nt: int, nx: int):
        """
        Internal method to generate the Saltelli sequence.
        
        :param nt: Number of base samples.
        :param nx: Input dimensions of sampled points.
        :return: A 2D array of Saltelli sequence samples.
        """
        N = nt
        nInput = nx
        skipValue = self.skipValue
        calSecondOrder = self.calSecondOrder
        
        M = None
        if skipValue > 0 and isinstance(skipValue, int):
            M = skipValue
            if not ((M & (M - 1)) == 0 and (M != 0 and M - 1 != 0)):
                raise ValueError("skip value must be a power of 2!")
            if N < M:
                raise ValueError("N must be greater than skip value you set!")
            
        elif skipValue < 0 or not isinstance(skipValue, int):
            raise ValueError("skip value must be a positive integer!")
        
        sampler = qmc.Sobol(nInput * 2, scramble=self.scramble, seed=1)
        
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