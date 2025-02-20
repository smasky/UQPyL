import numpy as np
from typing import Optional, Tuple

from ..surrogates.mars import MARS
from .saABC import SA
from ..utility import MinMaxScaler, Scaler, Verbose
from ..problems import ProblemABC as Problem
from ..DoE import LHS, Sampler

class MARS_SA(SA):
    '''
    -------------------------------------------------
    Multivariate Adaptive Regression Splines for Sensibility Analysis
    -------------------------------------------------
    This class implements the MARS method, which is 
    used for sensitivity analysis of model outputs.
    
    Methods:
        sample: Generate a sample for MARS analysis
        analyze: Perform MARS analysis from the X and Y you provided.
    
    Examples:
        # `problem` is an instance of ProblemABC or Problem from UQPyL.problems
        #  You must create a problem instance before using this method.
        >>> mars_method = MARS_SA()
        >>> X = mars_method.sample(problem, 500)
        >>> Y = problem.evaluate(X)
        >>> res = mars_method.analyze(problem, X, Y)
        >>> print(res)
        
    References:
        [1] J. H. Friedman, Multivariate Adaptive Regression Splines, 
            The Annals of Statistics, vol. 19, no. 1, pp. 1-67, Mar. 1991, 
            doi: 10.1214/aos/1176347963.
        [2] SALib, https://github.com/SALib/SALib
    --------------------------------------------------------------------------
    '''
    
    name = "MARS_SA"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        '''
        Initialize the MARS_SA method.
        
        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data. Defaults to (None, None).
        :param verboseFlag: bool - If True, enables verbose mode for logging. Defaults to False.
        :param logFlag: bool - If True, saves logging to a file. Defaults to False.
        :param saveFlag: bool - If True, saves the results to a file. Defaults to False.
        '''
        
        # Attribute indicating the types of sensitivity indices calculated
        self.firstOrder = True
        self.secondOrder = False
        self.totalOrder = False
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
    
    def sample(self, problem: Problem, N: int = 500, sampler: Sampler = LHS('classic')):
        '''
        Generate a sample set for the MARS method.

        :param problem: Problem - The problem instance defining the input space.
        :param N: int, optional - The number of samples to generate. Defaults to 500.
        :param sampler: Sampler, optional - The sampling method to use. Defaults to Latin Hypercube Sampling (LHS) with 'classic' mode.

        :return: np.ndarray - A 2D array of shape `(N, nInput)`, where `nInput` is the number of input variables.
        '''
        
        nInput = problem.nInput
        
        # Generate samples using the specified sampler
        X = sampler.sample(N, nInput)
        
        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray = None, Y: np.ndarray = None):
        '''
        Perform the MARS analysis on the input data.

        :param problem: Problem - The problem instance that defines the input and output space.
        :param X: np.ndarray - A 2D array of shape `(N, nInput)`, representing the input data for analysis.
        :param Y: np.ndarray, optional - A 1D array of length `N` representing the output values corresponding to `X`. 
                  If None, it will be computed by evaluating the problem with `X`.

        :return: Result - An object containing the sensitivity indices 'S1', 'S2', and 'ST', 
                          representing first-order, second-order, and total-order indices. 
                          You can use result.Si to get the sensitivity indices.
        '''
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        # Evaluate the problem if Y is not provided
        if Y is None:
            Y = self.evaluate(X)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        nInput = problem.nInput
        
        # Initialize an array to store first-order sensitivity indices
        S1 = np.zeros(nInput)
        
        # Main process: Fit the MARS model and calculate sensitivity indices
        mars = MARS(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)))
        mars.fit(X, Y)
        base_gcv = mars.gcv_
        
        # Calculate first-order sensitivity indices for each input variable
        for i in range(nInput):
            X_sub = np.delete(X, [i], axis=1)
            mars = MARS(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)))
            mars.fit(X_sub, Y)
            S1[i] = np.abs(base_gcv - mars.gcv_)
            
        # Normalize the sensitivity indices
        S1_sum = sum(S1)
        S1 /= S1_sum
        
        # Record the calculated sensitivity indices
        self.record('S1', problem.xLabels, S1)
        
        # Return the result object containing all sensitivity indices
        return self.result