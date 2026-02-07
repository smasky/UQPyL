import numpy as np
from typing import Optional, Tuple


from .base import AnalysisABC
from ..util import MinMaxScaler, Scaler, Verbose
from ..problem import ProblemABC as Problem

from ..doe import LHS, Sampler
from ..surrogate.mars import MARS as MARSModel

class MARS(AnalysisABC):
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
    
    name = "MARS"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        '''
        Initialize the MARS_SA method.
        
        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data. Defaults to (None, None).
        :param verboseFlag: bool - If True, enables verbose mode for logging. Defaults to False.
        :param logFlag: bool - If True, saves logging to a file. Defaults to False.
        :param saveFlag: bool - If True, saves the results to a file. Defaults to False.
        '''
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
    
    def sample(self, problem: Problem, N: int = 500, sampler: Sampler = LHS('classic'), seed: Optional[int] = None):
        '''
        Generate a sample set for the MARS method.

        :param problem: Problem - The problem instance defining the input space.
        :param N: int, optional - The number of samples to generate. Defaults to 500.
        :param sampler: Sampler, optional - The sampling method to use. Defaults to Latin Hypercube Sampling (LHS) with 'classic' mode.

        :return: np.ndarray - A 2D array of shape `(N, nInput)`, where `nInput` is the number of input variables.
        '''
        
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        nInput = problem.nInput
        
        # Generate samples using the specified sampler
        sampler_seed = self.rng.integers(1, 1000000)
        X = sampler.sample(problem, N, seed = sampler_seed)
        
        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
    
    @Verbose.analyze
    def analyze(self, problem: Problem, X, Y: np.ndarray = None, target = 'objFunc', index = 'all'):
        '''
        Perform the MARS analysis on the input data.

        :param problem: Problem - The problem instance that defines the input and output space.
        :param X: np.ndarray - A 2D array of shape `(N, nInput)`, representing the input data for analysis.
        :param Y: np.ndarray, optional - A 1D array of length `N` representing the output values corresponding to `X`. 
                  If None, it will be computed by evaluating the problem with `X`.
        :param target: str - The target of the analysis, set 'objFunc' or 'conFunc'. Defaults to 'objFunc'.
        :param index: list - The index of the output variables to analyze. Defaults to 'all'.
        
        :return: Result - An object containing the sensitivity indices 'S1', 'S2', and 'ST', 
                          representing first-order, second-order, and total-order indices. 
                          You can use result.Si to get the sensitivity indices.
        '''
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)
        numY = Y.shape[1]
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        nInput = problem.nInput
        
        outputLabel = "obj" if target == "objFunc" else "con"
        
        S1 = np.zeros((numY, nInput))
        S1_scaled = np.zeros((numY, nInput))
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            
            Y_i = Y[:, i:i+1]
        
            # Main process: Fit the MARS model and calculate sensitivity indices
            mars = MARSModel(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)))
            mars.fit(X, Y_i)
            base_gcv = mars.gcv_

            # Calculate first-order sensitivity indices for each input variable
            for j in range(nInput):
                X_sub = np.delete(X, [j], axis=1)
                mars = MARSModel(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)))
                mars.fit(X_sub, Y_i)
                S1[i, j] = np.abs(base_gcv - mars.gcv_)
            
            S1_scaled[i] = S1[i] / np.sum(S1[i])
        
        res = [('S1', S1, row_label, col_label_1, 'decsDim1'), ('S1_scale', S1_scaled, row_label, col_label_1, 'decsDim1')]
        
        X, Y = self.__reverse_X_Y__(X, Y)
        
        self.recordResult(X, Y, res)
        
        return self.result.generateNetCDF()