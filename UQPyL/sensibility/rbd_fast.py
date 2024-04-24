import numpy as np
from scipy.signal import periodogram
from typing import Optional, Tuple

from .sa_ABC import SA
from ..DoE import Sampler, LHS
from ..problems import ProblemABC as Problem
from ..utility import Scaler
class RBD_FAST(SA):
    '''
        Random Balance Designs Fourier Amplitude Sensitivity Test
        ------------------------------------------------
        Parameters:
            problem: Problem
                The problem you want to analyse
            scaler: Tuple[Scaler, Scaler], default=(None, None)
                Used for scaling X or Y
            M: int, default=4
                The interference parameter, i.e., the number of harmonics to sum in the
                Fourier series decomposition (defalut 4). 
                But, the number of sample must be greater than 4*M**2!

            Following parameters derived from the variable 'problem'
            n_input: the input number of the problem
            ub: the upper bound of the problem
            lb: the lower bound of the problem
            
        Methods:
            sample: Generate a sample for RBD-FAST analysis
            analyze: perform RBD-FAST analyze from the X and Y you provided.
        
        Examples:
            >>> rbd_method=RBD_FAST(problem)
            >>> X=rbd_method.sample(500)
            >>> Y=problem.evaluate(X)
            >>> rbd_method.analyze(X, Y)
            
        Reference:
            [1] S. Tarantola et al, Random balance designs for the estimation of first order global sensitivity indices, 
                                    Reliability Engineering & System Safety, vol. 91, no. 6, pp. 717-727, Jun. 2006,
                                    doi: 10.1016/j.ress.2005.06.003.
            [2] J.-Y. Tissot and C. Prieur, Bias correction for the estimation of sensitivity indices based on random balance designs,
                                    Reliability Engineering & System Safety, vol. 107, pp. 205-213, Nov. 2012, 
                                    doi: 10.1016/j.ress.2012.06.010.
    '''
    def __init__(self, problem: Problem,scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), 
                       M: int=10):
        
        super().__init__(problem, scalers)
        self.M=M
    
    def sample(self, N: int=500, sampler: Sampler=LHS('classic')):
        '''
            Generate samples
            -------------------------------
            Parameter:
                N: int, default=500
                    N is corresponding to the use sampler 
                sampler: Sampler, default=LHS('classic')
            
            Returns:
                X: 2d-np.ndarray
                    the size is determined by the used sampler. Default: (N, n_input)            
        '''
        n_input=self.n_input
            
        X=sampler.sample(N, n_input)

        if N<=4*self.M**2:
            raise ValueError("the number of sample must be greater than 4*M**2!")
        
        return X
    
    def analyze(self, X: np.ndarray=None, Y: np.ndarray=None, verbose: bool=False):
        '''
            Perform RBD_FAST analysis
            -------------------------------------------------
            Parameters:
                X: np.ndarray
                    the input data
                Y: np.ndarray
                    the result data
                verbose: bool
                    the switch to print analysis summary or not     
            Returns:
                Si: dict
                    The type of Si is dict. And it contain 'S1' key value. 
        '''
        
        X, Y=self.__check_and_scale_xy__(X, Y)
        
        S1=np.zeros(n_input); n_input=self.n_input
        
        for i in range(n_input):
            idx=np.argsort(X[:, i])
            idx=np.concatenate([idx[::2], idx[1::2][::-1]])
            Y_seq=Y[idx]
            
            _, Pxx = periodogram(Y_seq.ravel())
            V=np.sum(Pxx[1:])
            D1=np.sum(Pxx[1: self.M+1])
            S1_sub=D1/V
            
            #####normalization
            lamb=(2*self.M)/Y.shape[0]
            S1_sub=S1_sub-lamb/(1-lamb)*(1-S1_sub)
            #####
            
            S1[i]=S1_sub
        
        Si={'S1':S1}
        self.Si=Si
        
        if verbose:
            self.summary()
            
        return Si
    
    def summary(self):
        '''
            print analysis summary
        '''
        if self.Si is None:
            raise ValueError("The sensitivity indices have not been performed yet!")
        
        print("Random Balance Designs Fourier Amplitude Sensitivity Test")
        print("-------------------------------------------------")
        print("Input Dimension: %d" % self.n_input)
        print("-------------------------------------------------")
        print("First Order Sensitivity Indices: ")
        print("-------------------------------------------------")
        for label, value in zip(self.x_labels, self.Si['S1']):
            print(f"{label}: {value:.4f}")
        print("-------------------------------------------------")
        print("-------------------------------------------------")
    
    
    def _default_sample(self):
        return self.sample(500)       
        
        