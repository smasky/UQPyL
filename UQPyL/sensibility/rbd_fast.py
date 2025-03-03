import numpy as np
from scipy.signal import periodogram
from typing import Optional, Tuple

from .saABC import SA
from ..DoE import Sampler, LHS
from ..problems import ProblemABC as Problem
from ..utility import Scaler, Verbose

class RBD_FAST(SA):
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
    
    name = "RBD_FAST"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None), 
                 M: int = 4, 
                 verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
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
        self.setParameters("M", M)
    
    def sample(self, problem: Problem, N: int = 500, M: Optional[int] = None, sampler: Sampler = LHS('classic')) -> np.ndarray:
        """
        Generate samples for RBD-FAST analysis.
        ---------------------------------------
        :param problem: Problem - The problem instance defining the input space.
        :param N: int, optional - The number of sample points. Defaults to 500.
        :param M: int, optional - The interference parameter. If None, uses the initialized value of M.
        :param sampler: Sampler, optional - The sampling strategy to use. Defaults to LHS with 'classic' method.

        :return: np.ndarray - A 2D array representing the generated sample points.
        """
        
        # Use the initialized value of M if not provided
        if M is None:
            M = self.getParaValue('M')
        else:
            self.setParameters('M', M)
            
        nInput = problem.nInput
        
        # Ensure the number of samples is sufficient
        if N <= 4 * M**2:
            raise ValueError("The number of sample must be greater than 4*M**2!")
        
        # Generate samples using the specified sampler
        X = sampler.sample(N, nInput)

        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: np.ndarray = None) -> dict:
        """
        Perform RBD-FAST analysis.
        ---------------------------------------
        :param problem: Problem - The problem instance defining the input and output space.
        :param X: np.ndarray - A 2D array representing the input data for analysis.
        :param Y: np.ndarray, optional - A 1D array representing the output values corresponding to `X`. If None, it will be computed by evaluating the problem with `X`.

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
        if Y is None:
            Y = self.evaluate(X)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        
        # Initialize an array to store first-order sensitivity indices
        S1 = np.zeros(nInput)
        
        # Calculate sensitivity indices for each input variable
        for i in range(nInput):
            idx = np.argsort(X[:, i])
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
            
            S1[i] = S1_sub
        
        # Record the calculated sensitivity indices
        self.record('S1', problem.xLabels, S1)
        
        # Return the result object containing all sensitivity indices
        return self.result