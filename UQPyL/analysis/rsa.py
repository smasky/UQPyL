import numpy as np
from typing import Optional, Tuple
from scipy.stats import cramervonmises_2samp

from .base import AnalysisABC
from ..doe import LHS, Sampler
from ..problem import ProblemABC as Problem
from ..util import Scaler, Verbose

class RSA(AnalysisABC):
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
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the RSA method for sensitivity analysis.
        
        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data. Defaults to (None, None).
        :param verboseFlag: bool - If True, enables verbose mode for logging, providing detailed output during execution. Defaults to False.
        :param logFlag: bool - If True, enables logging of results to a file or console. Defaults to False.
        :param saveFlag: bool - If True, saves the results to a file for later analysis. Defaults to False.
        """
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
    
    def sample(self, problem: Problem, N: int, sampler: Sampler = LHS('classic'), seed: Optional[int] = None):
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
        
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        nInput = problem.nInput
        
        # Generate samples using the specified sampler
        sampler_seed = self.rng.integers(1, 1000000)
        X = sampler.sample(problem, N, seed = sampler_seed)
        
        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
        
    @Verbose.analyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: np.ndarray = None, target = 'objFunc', index = 'all', nRegion: int = 20):
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
        :param target: str - The target of the analysis, set 'objFunc' or 'conFunc'. Defaults to 'objFunc'.
        :param index: list - The index of the output variables to analyze. Defaults to 'all'.
        :param nRegion: int - The number of regions to divide the input space into. This affects the resolution of the sensitivity analysis. Defaults to 20.
        
        :return: Result - An object containing the sensitivity indices 'S1', 'S2', and 'ST', 
                          representing first-order, second-order, and total-order indices. 
                          You can use result.Si to get the sensitivity indices.
        """
        
        self.setParaValue("nRegion", nRegion)
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        nInput = problem.nInput
        
        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        
        numY = Y.shape[1]
        outputLabel = "obj" if target == "objFunc" else "con"
        
        S1 = np.zeros((numY, nInput))
        S1_scaled = np.zeros((numY, nInput))
        
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            
            Y_i = Y[:, i:i+1]
        
            # Define the sequence for dividing the input space into regions
            seq = np.linspace(0.0, 1.0, nRegion + 1)
            results = np.full((nRegion, nInput), np.nan)
            X_di = np.empty(X.shape[0])
            
            trr = Y_i.ravel()
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
            
            S1[i] = results_star
            S1_scaled[i] = results_star / np.sum(results_star)
        
        res = [('S1', S1, row_label, col_label_1, 'decsDim1'), ('S1_scale', S1_scaled, row_label, col_label_1, 'decsDim1')]
        
        X, Y = self.__reverse_X_Y__(X, Y)
        
        self.recordResult(X, Y, res)
        
        return self.result.generateNetCDF()
    
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