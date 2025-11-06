# Extend Fourier amplitude sensitivity test, FAST
import numpy as np
from typing import Optional, Tuple

from .base import AnalysisABC
from ..problem import ProblemABC as Problem
from ..util import Scaler, Verbose

class FAST(AnalysisABC):
    
    """
    -------------------------------------------------
    Fourier Amplitude Sensitivity Test (FAST)
    -------------------------------------------------
    This class implements the FAST method, which is 
    used for global sensitivity analysis of model outputs.
    
    Methods:
        sample: Generate a sample for FAST analysis
        analyze: Perform FAST analysis from the X and Y you provided.
    
    Examples:
        # `problem` is an instance of ProblemABC or Problem from UQPyL.problems
        #  You must create a problem instance before using this method.
        >>> fast_method = FAST()
        >>> X = fast_method.sample(problem)
        >>> res = fast_method.analyze(problem, X)
        >>> print(res)
        
    References:
        [1] Cukier et al., A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output,
            Technometrics, 41(1):39-56, doi: 10.1063/1.1680571
        [2] A. Saltelli et al., A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output,
            Technometrics, vol. 41, no. 1, pp. 39-56, Feb. 1999, doi: 10.1080/00401706.1999.10485594.
        [3] SALib, https://github.com/SALib/SALib
    --------------------------------------------------------------------------
    """
    
    name = "FAST"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the FAST method.
        ----------------------------------------------------------------
        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data. Defaults to (None, None).
        :param verboseFlag: bool - If True, enables verbose mode for logging. Defaults to False.
        :param logFlag: bool - If True, enables logging of results. Defaults to False.
        :param saveFlag: bool - If True, saves the results to a file. Defaults to False.
        """
            
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
        
        # Set the parameter for the number of harmonics
        self.setParaValue("M", 4)

    def sample(self, problem: Problem, N: Optional[int] = 500, M: Optional[int] = 4, seed: Optional[int] = None):
        """
        Generate a sample set for the FAST method.
        ----------------------------------------------------------------
        :param problem: Problem - The problem instance defining the input space.
        :param N: int, optional - The number of sample points for each sequence. Defaults to 500.
        :param M: int, optional - The Fourier frequency. If None, uses the initialized value of M.

        :return: np.ndarray - A 2D array representing the generated sample points.
        """
        
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        # Use the initialized value of M if not provided
        self.setParaValue("M", M)
        nInput = problem.nInput
        
        # Ensure the number of samples is sufficient
        if N < 4 * M**2:
            raise ValueError("The number of sample must be greater than 4*M**2! \n Default M = 4 .")
        
        # Initialize frequency array
        w = np.zeros(nInput)
        w[0] = np.floor((N - 1) / (2 * M))
        max_wi = np.floor(w[0] / (2 * M))
        
        # Assign frequencies to input variables
        if max_wi >= nInput - 1:
            w[1:] = np.floor(np.linspace(1, max_wi, nInput - 1))
        else:
            w[1:] = np.arange(nInput - 1) % max_wi + 1
        
        # Generate the sample points
        s = (2 * np.pi / N) * np.arange(N)
        
        X = np.zeros((N * nInput, nInput))
        w_tmp = np.zeros(nInput)
        
        for i in range(nInput):
            w_tmp[i] = w[0]
            idx = list(range(i)) + list(range(i + 1, nInput))
            w_tmp[idx] = w[1:]
            idx = range(i * N, (i + 1) * N)   
            phi = 2 * np.pi * self.rng.random()    
            sin_result = np.sin(w_tmp[:, None] * s + phi)
            arsin_result = (1 / np.pi) * np.arcsin(sin_result)  # Saltelli formula
            X[idx, :] = 0.5 + arsin_result.transpose()
        
        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
    
    @Verbose.analyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None, target = 'objFunc', index = 'all'):
        """
        Perform the FAST analysis on the input data.
        ----------------------------------------------------------------
        :param problem: Problem - The problem instance that defines the input and output space.
        :param X: np.ndarray - A 2D array of shape `(N * nInput, nInput)`, representing the input data for analysis.
        :param Y: np.ndarray, optional - A 1D array of length `N` representing the output values corresponding to `X`. 
                  If None, it will be computed by evaluating the problem with `X`.
        :param target: str - The target of the analysis, set 'objFunc' or 'conFunc'. Defaults to 'objFunc'.
        :param index: list - The index of the output variables to analyze. Defaults to 'all'.

        :return: Result - An object containing result of the analysis.
        """
        
        # Retrieve the parameter for the number of harmonics
        M = self.getParaValue('M')
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        self.check_Y(X, Y, target, index)
        numY = Y.shape[1]
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        
        nInput = problem.nInput
        n = int(X.shape[0] / nInput)
        
        # Initialize arrays to store sensitivity indices
        outputLabel = "obj" if target == "objFunc" else "con"
        
        # Calculate the base frequency
        w_0 = np.floor((n - 1) / (2 * M))
        
        S1 = np.zeros((numY, nInput))
        ST = np.zeros((numY, nInput))
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):

            Y_i = Y[:, i:i+1]
  
            # Calculate sensitivity indices for each input variable
            for j in range(nInput):
                idx = np.arange(j * n, (j + 1) * n)
                Y_sub = Y_i[idx]
                f = np.fft.fft(Y_sub.ravel())
                Sp = np.power(np.absolute(f[np.arange(1, np.ceil(n / 2), dtype=np.int32)]) / n, 2)
                V = 2.0 * np.sum(Sp)
                Di = 2.0 * np.sum(Sp[np.int32(np.arange(1, M + 1, dtype=np.int32) * w_0 - 1)])  # pw <= (NS-1)/2 w_0 = (NS-1)/M
                Dt = 2.0 * np.sum(Sp[np.arange(np.floor(w_0 / 2.0), dtype=np.int32)])
                
                S1[i, j] = Di / V
                ST[i, j] = 1.0 - Dt / V
                
        res = [('S1', S1, row_label, col_label_1, 'decsDim1'), ('ST', ST, row_label, col_label_1, 'decsDim1')]
        
        X, Y = self.__reverse_X_Y__(X, Y)
        
        self.recordResult(X, Y, res)
        
        # Return the result object containing all sensitivity indices
        return self.result.generateNetCDF()