import numpy as np
from typing import Optional, Tuple

from ..surrogates import MARS, Surrogate
from ..utility import MinMaxScaler, Scaler
from ..problems import ProblemABC as Problem
from ..DoE import LHS, Sampler
from .sa_ABC import SA

class MARS_SA(SA):
    '''
        Multivariate Adaptive Regression Splines - Sensibility Analysis
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
    def __init__(self, problem: Problem, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None)):
        
        super().__init__(problem, scalers)
    
    def sample(self, N: int=500, sampler: Sampler=LHS('classic')):
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
        n_input=self.n_input
        
        X=sampler.sample(N, n_input)
        
        return X
    
    def analyze(self, X: np.ndarray=None, Y: np.ndarray=None, verbose: bool=False):
        '''
            Perform MARS-SA
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
        X, Y=self.__check_and_scale_xy__(X, Y)
        n_input=self.n_input
        
        S1=np.zeros(n_input)
        #main process    
        mars=MARS(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)))
        mars.fit(X, Y)
        base_gcv=mars.gcv_
        
        for i in range(n_input):
            X_sub=np.delete(X, [i], axis=1)
            mars=MARS(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)))
            mars.fit(X_sub, Y)
            S1[i]=np.abs(base_gcv-mars.gcv_)
            
        S1_sum = sum(S1)
        S1/=S1_sum
        
        Si={'S1': S1}
        self.Si=Si
        
        if verbose:
            self.summary()
        
        return S1
    
    def summary(self):
        '''
            print summary analysis
        '''
        
        print('MARS Sensibility Analysis')
        print("-------------------------------------------------")
        print("Input Dimension: %d" % self.n_input)
        print("-------------------------------------------------")
        print("S1 value:")
        for label, value in zip(self.x_labels, self.Si['S1']):
            print(f"{label}: {value:.4f}")
        print("-------------------------------------------------")
        
        
        
        
        
        
        
        
        
        
        