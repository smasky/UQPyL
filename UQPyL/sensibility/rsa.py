import numpy as np
from typing import Optional, Tuple
from scipy.stats import cramervonmises_2samp

from .saABC import SA
from ..DoE import LHS, Sampler
from ..problems import ProblemABC as Problem
from ..utility import Scaler, Verbose

class RSA(SA):
    '''
    -------------------------------------------------
    Regional Sensitivity Analysis (RSA)
    -------------------------------------------------
    This class implements the RSA method, which is used for 
    sensitivity analysis by dividing the input space into regions 
    and analyzing the influence of input factors on model outputs.

    Parameters:
        problem (Problem): 
            The problem instance defining the input space.
        n_region (int, optional): 
            The number of regions to divide the input space into. 
            This determines the granularity of the sensitivity analysis. 
            Defaults to 20.
        scalers (Tuple[Scaler, Scaler], optional): 
            Tuple containing scalers for input (X) and output (Y) data. 
            Defaults to (None, None), meaning no scaling is applied.
        verboseFlag (bool): 
            If True, enables verbose mode for logging. Defaults to False.
        logFlag (bool): 
            If True, enables logging of results. Defaults to False.
        saveFlag (bool): 
            If True, saves the results to a file. Defaults to False.

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
    '''
    
    name="RSA"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 nRegion: int=20,
                 verboseFlag: bool=False, logFlag: bool=False, saveFlag: bool=False):
        '''
        Initialize the RSA method for sensitivity analysis.
        
        The RSA method divides the input space into regions and evaluates 
        the sensitivity of model outputs to variations in input factors 
        within these regions.

        Parameters:
            scalers (Tuple[Optional[Scaler], Optional[Scaler]]): 
                Tuple containing scalers for input (X) and output (Y) data. 
                Defaults to (None, None), meaning no scaling is applied.
            nRegion (int): 
                The number of regions to divide the input space into. 
                This affects the resolution of the sensitivity analysis. 
                Defaults to 20.
            verboseFlag (bool): 
                If True, enables verbose mode for logging, providing detailed 
                output during execution. Defaults to False.
            logFlag (bool): 
                If True, enables logging of results to a file or console. 
                Defaults to False.
            saveFlag (bool): 
                If True, saves the results to a file for later analysis. 
                Defaults to False.
        '''
        
        #Attribute
        self.firstOrder=True
        self.secondOrder=False
        self.totalOrder=False
        
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)

        self.setParameters("nRegion", nRegion)
    
    def sample(self, problem: Problem, N: int, sampler: Sampler=LHS('classic')):
        '''
        Generate samples for RSA analysis
        ---------------------------------------
        This method generates a sample of input data `X` using a specified 
        sampling strategy, typically Latin Hypercube Sampling (LHS), for 
        the RSA method.

        Parameters:
            problem (Problem): 
                The problem instance defining the input space.
            N (int): 
                The number of sample points to generate.
            sampler (Sampler, optional): 
                The sampling strategy to use. Defaults to LHS with 'classic' method.

        Returns:
            np.ndarray: 
                A 2D array representing the generated sample points, with shape `(N, nInput)`.
        '''
        
        nInput=problem.nInput
        
        X=sampler.sample(N, nInput)
        
        return problem._transform_unit_X(X)
        
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: np.ndarray=None):
        '''
        Perform RSA analysis
        -------------------------------------
        This method performs the RSA sensitivity analysis by dividing the 
        input space into regions and evaluating the influence of input 
        factors on model outputs within these regions.

        Parameters:
            problem (Problem): 
                The problem instance defining the input and output space.
            X (np.ndarray): 
                A 2D array representing the input data for analysis.
            Y (np.ndarray, optional): 
                A 1D array representing the output values corresponding to `X`. 
                If None, it will be computed by evaluating the problem with `X`.

        Returns:
            dict: 
                A dictionary containing the sensitivity index 'S1', which 
                represents the first-order sensitivity indices for each input factor.
        '''
        
        nRegion=self.getParaValue("nRegion")
        
        self.setProblem(problem)
        
        nInput=problem.nInput
        
        if Y is None:
            Y=self.evaluate(X)
        
        X, Y=self.__check_and_scale_xy__(X, Y)
        
        seq = np.linspace(0.0, 1.0, nRegion + 1)
        results = np.full((nRegion, nInput), np.nan)
        X_di = np.empty(X.shape[0])
        
        trr=Y.ravel()
        mrr=X_di
        
        for d_i in range(nInput):
            X_di[:] = X[:, d_i]
            
            quants=np.quantile(trr, seq)
            
            b = (quants[0] <= trr) & (trr <= quants[1])
            if self._has_samples(Y, b):
                results[0, d_i] = cramervonmises_2samp(mrr[b].ravel(), mrr[~b].ravel()).statistic
             
            for bin_index in range(1, nRegion):
                
                b = (quants[bin_index] < trr) & (trr <= quants[bin_index+1])
                
                if self._has_samples(Y, b):
                    results[bin_index, d_i] = cramervonmises_2samp(mrr[b].ravel(), mrr[~b].ravel()).statistic
        
        results_star = np.mean(results, axis=0)
        self.record("S1", problem.xLabels, results_star)
        self.record("S1(Scaled)", problem.xLabels, results_star/np.sum(results_star))
        
        return self.result
    
    def _has_samples(self, y, sel):
        '''
        Check if the selected samples are sufficient for analysis.

        This helper method ensures that the selected samples are non-empty 
        and contain enough unique values for meaningful statistical analysis.

        Parameters:
            y (np.ndarray): 
                The output data array.
            sel (np.ndarray): 
                A boolean array indicating the selected samples.

        Returns:
            bool: 
                True if the selected samples are sufficient, False otherwise.
        '''
        return(
            (np.count_nonzero(sel) !=0)
             and (len(y[~sel])!=0 )
             and np.unique(y[sel]).size > 1
        )