import numpy as np
from typing import Optional, Tuple
from scipy.stats import cramervonmises_2samp

from .saABC import SA
from ..DoE import LHS, Sampler
from ..problems import ProblemABC as Problem
from ..utility import Scaler, Verbose

class RSA(SA):
    '''
        Regonal Sensitivity Analysis
        ---------------------------
        Parameters:
            problem: Problem
                the problem you want to analyse
            n_region: int, default=20
                the number of region you want to divide
            scaler: Tuple[Scaler, Scaler], default=(None, None)
                used for scaling X or Y
             
            Following parameters derived from the variable 'problem'
            n_input: the input number of the problem
            ub: the upper bound of the problem
            lb: the lower bound of the problem
        
        Methods:
            sample: Generate a sample for RSA analysis
            analyze: perform RSA analyze from the X and Y you provided.
        
        Examples:
            >>>rsa_method=RSA(problem)
            >>>X=rsa_method.sample(500)
            >>>Y=problem.evaluate(X)
            >>>Si=rsa_method.analyze(X, Y)
        
        References:
            [1] F. Pianosi et al., Sensitivity analysis of environmental models: A systematic review with practical workflow, 
                                   Environmental Modelling & Software, vol. 79, pp. 214-232, May 2016, 
                                   doi: 10.1016/j.envsoft.2016.02.008.
            [2] SALib, https://github.com/SALib/SALib
    '''
    name="RSA"
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 nRegion: int=20, N: int=500,
                 verbose: bool=False, logFlag: bool=False, saveFlag: bool=False):
        
        #Attribute
        self.firstOrder=True
        self.secondOrder=False
        self.totalOrder=False
        
        super().__init__(scalers, verbose, logFlag, saveFlag)

        self.setParameters("nRegion", nRegion)
        self.setParameters("N", N)
    
    def sample(self, problem: Problem, N: Optional[int]=None, sampler: Sampler=LHS('classic')):
        '''
            Generate samples
            -------------------------------
            Parameters:
                N: int, default=500
                    N is corresponding to the use sampler 
                sampler: Sampler, default=LHS('classic')
            
            Returns:
                X: 2d-np.ndarray
                    the size is determined by the used sampler. Default: (N, n_input)            
        '''
        if N is None:
            N=self.getParaValue("N")
        
        self.setParameters("N", N)
        
        nInput=problem.nInput
        
        X=sampler.sample(N, nInput)
        
        return X
        
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray=None, Y: np.ndarray=None):
        '''
            Perform RSA
            -------------------------------------
            Parameters:
                X: np.ndarray
                    the input data
                Y: np.ndarray
                    the result data
                verbose: bool 
                    the switch to print analysis summary or not
            
            Returns:
                Si: dict
                    The type of Si is dict. It contains 'S1'.
        '''
        N, nRegion=self.getParaValue("N", "nRegion")
        self.setProblem(problem)
        nInput=problem.nInput
        
        if X is None or Y is None:
            X=self.sample(problem, N)
            Y=problem.evaluate(X)
        
        X, Y=self.__check_and_scale_xy__(X, Y)
        
        seq = np.linspace(0, 1, nRegion + 1)
        results = np.full((nRegion, nInput), np.nan)
        X_di = np.empty(X.shape[0])
        
        for d_i in range(nInput):
                X_di = X[:, d_i]
                for bin_index in range(nRegion):
                    lower_bound, upper_bound = seq[bin_index], seq[bin_index + 1]
                    b = (lower_bound < X_di) & (X_di <= upper_bound)
                    if np.count_nonzero(b) > 0 and np.unique(X[b]).size > 1:
                        r_s = cramervonmises_2samp(Y[b].ravel(), Y[~b].ravel()).statistic
                        results[bin_index, d_i] = r_s

        self.record("S1", problem.x_labels, results)
        
        return self.result