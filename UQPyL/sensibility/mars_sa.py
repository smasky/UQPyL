import numpy as np
from typing import Optional, Tuple

from ..surrogates.mars import MARS
from .saABC import SA
from ..utility import MinMaxScaler, Scaler, Verbose
from ..problems import ProblemABC as Problem
from ..DoE import LHS, Sampler

class MARS_SA(SA):
    '''
        Multivariate Adaptive Regression Splines for Sensibility Analysis
        -------------------------------------------------------
        Parameters:
            Parameters:
                problem: Problem
                    the problem you want to analyse
                scaler: Tuple[Scaler, Scaler], default=(None, None)
                    used for scaling X or Y
        Methods:
            sample: Generate a sample for MARS analysis
            analyze: perform MARS analyze from the X and Y you provided.
        
        Examples:
            >>> mars_method=MARS_SA(problem)
            >>> X=mars_method.sample(500)
            >>> Y=problem.evaluate(X)
            >>> mars_method.analyze(X, Y)
        
        References:
            [1] J. H. Friedman, Multivariate Adaptive Regression Splines, 
                                The Annals of Statistics, vol. 19, no. 1, pp. 1-67, Mar. 1991, 
                                doi: 10.1214/aos/1176347963.
            [2] SALib, https://github.com/SALib/SALib
    '''
    
    name="MARS_SA"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        
        #Attribute
        self.firstOrder = True
        self.secondOrder = False
        self.totalOrder = False
        
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
    
    def sample(self, problem: Problem, N: int=500, sampler: Sampler = LHS('classic')):
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
        nInput = problem.nInput
        
        X = sampler.sample(N, nInput)
        
        return problem._transform_unit_X(X)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray = None, Y: np.ndarray = None):
        '''
            Perform MARS-SA
            -------------------------------------
            Parameters:
                X: np.ndarray
                    the input data
                Y: np.ndarray
                    the result data
            
            Returns:
                Si: dict
                    The type of Si is dict. It contains 'S1'.
        '''
        self.setProblem(problem)
        
        if Y is None:
            Y = self.evaluate(X)
        
        X, Y = self.__check_and_scale_xy__(X, Y)
        nInput = problem.nInput
        
        S1 = np.zeros(nInput)
        
        #main process    
        mars=MARS( scalers = (MinMaxScaler(0,1), MinMaxScaler(0,1)) )
        mars.fit(X, Y)
        base_gcv = mars.gcv_
        
        for i in range(nInput):
            X_sub = np.delete(X, [i], axis=1)
            mars = MARS( scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)) )
            mars.fit(X_sub, Y)
            S1[i] = np.abs(base_gcv - mars.gcv_)
            
        S1_sum = sum(S1)
        S1/=S1_sum
        
        self.record('S1', problem.xLabels, S1)
        
        return self.result

        
        
        
        
        
        
        
        
        
        
        