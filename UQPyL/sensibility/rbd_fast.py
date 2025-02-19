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

    Parameters:
        problem (Problem): 
            The problem instance defining the input space.
        scalers (Tuple[Scaler, Scaler], optional): 
            Tuple containing scalers for input (X) and output (Y) data. 
            Defaults to (None, None).
        M (int): 
            The interference parameter, i.e., the number of harmonics to sum in the
            Fourier series decomposition. Defaults to 4.
        verboseFlag (bool): 
            If True, enables verbose mode for logging. Defaults to False.
        logFlag (bool): 
            If True, enables logging of results. Defaults to False.
        saveFlag (bool): 
            If True, saves the results to a file. Defaults to False.

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
    
    name="RBD_FAST"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), 
                       M: int=4, 
                       verboseFlag: bool=False, logFlag: bool=False, saveFlag: bool=False):
        '''
        Initialize the RBD-FAST method for global sensitivity analysis.
        
        The RBD-FAST method uses random balance designs to estimate first-order 
        sensitivity indices, providing a robust approach to understanding the 
        influence of input factors on model outputs.

        Parameters:
            scalers (Tuple[Optional[Scaler], Optional[Scaler]]): 
                Tuple containing scalers for input (X) and output (Y) data. 
                Defaults to (None, None), meaning no scaling is applied.
            M (int): 
                The interference parameter, representing the number of harmonics 
                to sum in the Fourier series decomposition. This affects the 
                resolution of the sensitivity analysis. Defaults to 4.
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
        
        self.setParameters("M", M)
    
    def sample(self, problem: Problem, N: int=500, M: Optional[int]=None, sampler: Sampler=LHS('classic')):
        '''
        Generate samples for RBD-FAST analysis
        ---------------------------------------
        This method generates a sample of input data `X` using a specified 
        sampling strategy, typically Latin Hypercube Sampling (LHS), for 
        the RBD-FAST method.

        Parameters:
            problem (Problem): 
                The problem instance defining the input space.
            N (int, optional): 
                The number of sample points. Defaults to 500.
            M (int, optional): 
                The interference parameter. If None, uses the initialized value of M.
            sampler (Sampler, optional): 
                The sampling strategy to use. Defaults to LHS with 'classic' method.

        Returns:
            np.ndarray: 
                A 2D array representing the generated sample points.
        '''
        
        if M is None:
            M=self.getParaValue('M')
        else:
            self.setParameters('M', M)
            
        nInput=problem.nInput
        
        if N <= 4*M**2:
            raise ValueError("The number of sample must be greater than 4*M**2!")
        
        X=sampler.sample(N, nInput)

        return problem._transform_unit_X(X)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: np.ndarray=None):
        '''
        Perform RBD-FAST analysis
        -------------------------------------------------
        This method performs the RBD-FAST sensitivity analysis by estimating 
        the first-order sensitivity indices based on the provided input data 
        `X` and output data `Y`.

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
        
        M = self.getParaValue('M')
        
        self.setProblem(problem)
        
        nInput = problem.nInput
        
        if Y is None:
            Y=self.evaluate(X)
        
        X, Y=self.__check_and_scale_xy__(X, Y)
        
        S1=np.zeros(nInput)
        
        for i in range(nInput):
            idx=np.argsort(X[:, i])
            idx=np.concatenate([idx[::2], idx[1::2][::-1]])
            Y_seq=Y[idx]
            
            _, Pxx = periodogram(Y_seq.ravel())
            V=np.sum(Pxx[1:])
            D1=np.sum(Pxx[1: M+1])
            S1_sub=D1/V
            
            #####normalization
            lamb=(2*M)/Y.shape[0]
            S1_sub=S1_sub-lamb/(1-lamb)*(1-S1_sub)
            #####
            
            S1[i]=S1_sub
        
        self.record('S1', problem.xLabels, S1)
        
        return self.result