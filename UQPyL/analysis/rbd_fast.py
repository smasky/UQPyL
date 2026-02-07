import numpy as np
from scipy.signal import periodogram
from typing import Optional, Tuple

from .base import AnalysisABC
from ..doe import Sampler, LHS
from ..problem import ProblemABC as Problem
from ..util import Scaler, Verbose

class RBDFAST(AnalysisABC):
    """
    -------------------------------------------------
    Random Balance Designs Fourier Amplitude Sensitivity Test (RBD-FAST)
    -------------------------------------------------
    This class implements the RBD-FAST method, which is 
    used for global sensitivity analysis by estimating 
    first-order sensitivity indices using random balance designs.

    Methods:
        sample: Generate a sample for RBD-FAST analysis
        analyze: Perform RBD-FAST analysis from the X and Y you provided.

    Examples:
        # `problem` is an instance of ProblemABC or Problem from UQPyL.problems
        #  You must create a problem instance before using this method.
        >>> rbd_method = RBD_FAST(problem)
        >>> X = rbd_method.sample(500)
        >>> Y = problem.evaluate(X)
        >>> rbd_method.analyze(X, Y)

    References:
        [1] S. Tarantola et al, Random balance designs for the estimation of first order global sensitivity indices, 
            Reliability Engineering & System Safety, vol. 91, no. 6, pp. 717-727, Jun. 2006,
            doi: 10.1016/j.ress.2005.06.003.
        [2] J.-Y. Tissot and C. Prieur, Bias correction for the estimation of sensitivity indices based on random balance designs,
            Reliability Engineering & System Safety, vol. 107, pp. 205-213, Nov. 2012, 
            doi: 10.1016/j.ress.2012.06.010.
    -------------------------------------------------
    """
    
    name = "RBDFAST"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None), 
                 M: int = 4, 
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the RBD-FAST method for global sensitivity analysis.
        ----------------------------------------------------------------
        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data. Defaults to (None, None).
        :param M: int - The interference parameter, representing the number of harmonics to sum in the Fourier series decomposition. Defaults to 4.
        :param verboseFlag: bool - If True, enables verbose mode for logging. Defaults to False.
        :param logFlag: bool - If True, enables logging of results. Defaults to False.
        :param saveFlag: bool - If True, saves the results to a file. Defaults to False.
        """
        # Attribute indicating the types of sensitivity indices calculated
        self.firstOrder = True
        self.secondOrder = False
        self.totalOrder = False
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
        
        # Set the parameter for the number of harmonics
        self.setParaValue("M", 4)
    
    def sample(self, problem: Problem, N: int = 512, M: int = 4, sampler: Sampler = LHS('classic'), seed: Optional[int] = None):
        """
        Generate samples for RBD-FAST analysis.
        ---------------------------------------
        :param problem: Problem - The problem instance defining the input space.
        :param N: int, optional - The number of sample points. Defaults to 500.
        :param M: int, optional - The interference parameter. If None, uses the initialized value of M.
        :param sampler: Sampler, optional - The sampling strategy to use. Defaults to LHS with 'classic' method.

        :return: np.ndarray - A 2D array representing the generated sample points.
        """
        
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        self.setParaValue("M", M)
            
        nInput = problem.nInput
        
        # Ensure the number of samples is sufficient
        if N <= 4 * M**2:
            raise ValueError("The number of sample must be greater than 4*M**2!")
        
        # Generate samples using the specified sampler
        sampler_seed = self.rng.integers(1, 1000000)
        X = sampler.sample(problem, N, seed = sampler_seed)

        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
    
    @Verbose.analyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: np.ndarray = None, target = 'objFunc', index = 'all'):
        """
        Perform RBD-FAST analysis.
        ---------------------------------------
        :param problem: Problem - The problem instance defining the input and output space.
        :param X: np.ndarray - A 2D array representing the input data for analysis.
        :param Y: np.ndarray, optional - A 1D array representing the output values corresponding to `X`. If None, it will be computed by evaluating the problem with `X`.
        :param target: str - The target of the analysis, set 'objFunc' or 'conFunc'. Defaults to 'objFunc'.
        :param index: list - The index of the output variables to analyze. Defaults to 'all'.
        
        :return: Result - An object containing the sensitivity indices 'S1', 'S2', and 'ST', 
                          representing first-order, second-order, and total-order indices. 
                          You can use result.Si to get the sensitivity indices.
        """
        
        # Retrieve the parameter for the number of harmonics
        M = self.getParaValue('M')
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        nInput = problem.nInput
        
        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        
        numY = Y.shape[1]
        outputLabel = "obj" if target == "objFunc" else "con"
        
        # Initialize an array to store first-order sensitivity indices
        
        S1 = np.zeros((numY, nInput))
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            
            Y_i = Y[:, i:i+1]
            
            # Calculate sensitivity indices for each input variable
            for j in range(nInput):
                idx = np.argsort(X[:, j])
                idx = np.concatenate([idx[::2], idx[1::2][::-1]])
                Y_seq = Y[idx]
                
                # Perform periodogram analysis
                _, Pxx = periodogram(Y_seq.ravel())
                V = np.sum(Pxx[1:])
                D1 = np.sum(Pxx[1: M+1])
                S1_sub = D1 / V
                
                # Normalization
                lamb = (2 * M) / Y.shape[0]
                S1_sub = S1_sub - lamb / (1 - lamb) * (1 - S1_sub)
                
                S1[i, j] = S1_sub
        
        res = [('S1', S1, row_label, col_label_1, 'decsDim1')]
        
        X, Y = self.__reverse_X_Y__(X, Y)
        
        self.recordResult(X, Y, res)
        
        # Return the result object containing all sensitivity indices
        return self.result.generateNetCDF()