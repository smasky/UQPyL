import numpy as np
from typing import Optional, Tuple
from scipy.stats import cramervonmises_2samp

from .saABC import SA
from ..DoE import LHS, Sampler
from ..problems import ProblemABC as Problem
from ..utility import Scaler, Verbose

class RSA(SA):
    """
    -------------------------------------------------
    Regional Sensitivity Analysis (RSA)
    -------------------------------------------------
    This class implements the RSA method, which is used for 
    sensitivity analysis by dividing the input space into regions 
    and analyzing the influence of input factors on model outputs.

    Methods:
        sample: Generate a sample for RSA analysis
        analyze: Perform RSA analysis from the X and Y you provided.

    Examples:
        # `problem` is an instance of ProblemABC or Problem from UQPyL.problems
        #  You must create a problem instance before using this method.
        >>> rsa_method = RSA(problem)
        >>> X = rsa_method.sample(500)
        >>> Y = problem.evaluate(X)
        >>> Si = rsa_method.analyze(X, Y)

    References:
        [1] F. Pianosi et al., Sensitivity analysis of environmental models: A systematic review with practical workflow, 
            Environmental Modelling & Software, vol. 79, pp. 214-232, May 2016, 
            doi: 10.1016/j.envsoft.2016.02.008.
        [2] SALib, https://github.com/SALib/SALib
    -------------------------------------------------
    """
    
    name = "RSA"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 nRegion: int = 20,
                 verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the RSA method for sensitivity analysis.
        
        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data. Defaults to (None, None).
        :param nRegion: int - The number of regions to divide the input space into. This affects the resolution of the sensitivity analysis. Defaults to 20.
        :param verboseFlag: bool - If True, enables verbose mode for logging, providing detailed output during execution. Defaults to False.
        :param logFlag: bool - If True, enables logging of results to a file or console. Defaults to False.
        :param saveFlag: bool - If True, saves the results to a file for later analysis. Defaults to False.
        """
        
        # Attribute indicating the types of sensitivity indices calculated
        self.firstOrder = True
        self.secondOrder = False
        self.totalOrder = False
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)

        # Set the number of regions for analysis
        self.setParameters("nRegion", nRegion)
    
    def sample(self, problem: Problem, N: int, sampler: Sampler = LHS('classic')):
        """
        Generate samples for RSA analysis
        ---------------------------------------
        This method generates a sample of input data `X` using a specified 
        sampling strategy, typically Latin Hypercube Sampling (LHS), for 
        the RSA method.

        :param problem: Problem - The problem instance defining the input space.
        :param N: int - The number of sample points to generate.
        :param sampler: Sampler, optional - The sampling strategy to use. Defaults to LHS with 'classic' method.

        :return: np.ndarray - A 2D array representing the generated sample points, with shape `(N, nInput)`.
        """
        
        nInput = problem.nInput
        
        # Generate samples using the specified sampler
        X = sampler.sample(N, nInput)
        
        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
        
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: np.ndarray = None):
        """
        Perform RSA analysis
        -------------------------------------
        This method performs the RSA sensitivity analysis by dividing the 
        input space into regions and evaluating the influence of input 
        factors on model outputs within these regions.

        :param problem: Problem - The problem instance defining the input and output space.
        :param X: np.ndarray - A 2D array representing the input data for analysis.
        :param Y: np.ndarray, optional - A 1D array representing the output values corresponding to `X`. 
                  If None, it will be computed by evaluating the problem with `X`.

        :return: Result - An object containing the sensitivity indices 'S1', 'S2', and 'ST', 
                          representing first-order, second-order, and total-order indices. 
                          You can use result.Si to get the sensitivity indices.
        """
        
        # Retrieve the number of regions for analysis
        nRegion = self.getParaValue("nRegion")
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        nInput = problem.nInput
        
        # Evaluate the problem if Y is not provided
        if Y is None:
            Y = self.evaluate(X)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        
        # Define the sequence for dividing the input space into regions
        seq = np.linspace(0.0, 1.0, nRegion + 1)
        results = np.full((nRegion, nInput), np.nan)
        X_di = np.empty(X.shape[0])
        
        trr = Y.ravel()
        mrr = X_di
        
        # Loop over each input dimension to perform RSA
        for d_i in range(nInput):
            X_di[:] = X[:, d_i]
            
            # Calculate quantiles for dividing the output space
            quants = np.quantile(trr, seq)
            
            # Perform analysis for each region
            b = (quants[0] <= trr) & (trr <= quants[1])
            if self._has_samples(Y, b):
                results[0, d_i] = cramervonmises_2samp(mrr[b].ravel(), mrr[~b].ravel()).statistic
             
            for bin_index in range(1, nRegion):
                
                b = (quants[bin_index] < trr) & (trr <= quants[bin_index+1])
                
                if self._has_samples(Y, b):
                    results[bin_index, d_i] = cramervonmises_2samp(mrr[b].ravel(), mrr[~b].ravel()).statistic
        
        # Calculate the mean sensitivity index for each input factor
        results_star = np.mean(results, axis=0)
        self.record("S1", problem.xLabels, results_star)
        self.record("S1(Scaled)", problem.xLabels, results_star / np.sum(results_star))
        
        return self.result
    
    def _has_samples(self, y, sel):
        """
        Check if the selected samples are sufficient for analysis.

        This helper method ensures that the selected samples are non-empty 
        and contain enough unique values for meaningful statistical analysis.

        :param y: np.ndarray - The output data array.
        :param sel: np.ndarray - A boolean array indicating the selected samples.

        :return: bool - True if the selected samples are sufficient, False otherwise.
        """
        return (
            (np.count_nonzero(sel) != 0)
            and (len(y[~sel]) != 0)
            and np.unique(y[sel]).size > 1
        )