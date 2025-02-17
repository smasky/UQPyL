# Extend Fourier amplitude sensitivity test, FAST
import numpy as np
from typing import Optional, Tuple

from .saABC import SA
from ..problems import ProblemABC as Problem
from ..utility import Scaler, Verbose
class FAST(SA):
    
    name="FAST"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                    M: int =4,
                    verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        '''
        Fourier amplitude sensitivity test (FAST) or extend Fourier amplitude sensitivity test (eFAST)
        ---------------------------
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
            sample: Generate a sample for FAST analysis
            analyze: perform FAST analyze from the X and Y you provided.
        
        Examples:
            >>> fast_method=FAST(problem)
            >>> X=fast_method.sample()
        
        References:
            [1] Cukier et, al, A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output
                               Technometrics, 41(1):39-56,
                               doi: 10.1063/1.1680571
            [2] A. Saltelli et al, A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output,
                                   Technometrics, vol. 41, no. 1, pp. 39-56, Feb. 1999, 
                                   doi: 10.1080/00401706.1999.10485594.
            [3] SALib, https://github.com/SALib/SALib
        '''
        
        #Attribute
        self.firstOrder = True
        self.secondOrder = False
        self.totalOrder = True
    
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
        
        self.setParameters("M", M)

    def sample(self, problem: Problem, N: Optional[int] = 500, M: Optional[int] = None):
        '''
            -----------------------------------------------------
            Generate FAST sequence from paper [2]
            -----------------------------------------------------
            Parameters:
                N: int, default = 500
                    the number of sample points for each sequence
                M: int, default = 4
                    the fourier frequency

            Returns:
                X: np.ndarray -> 2d-array
                    the size of X is ( ( N*nInput, nInput ) )
            -----------------------------------------------------
        '''
        
        # 
        if M is None:
            M = self.getParaValue('M')
        else:
            self.setParameters("M", M)
        
        nInput = problem.nInput
        
        if N < 4*M**2:
            raise ValueError("The number of sample must be greater than 4*M**2! \n Default M = 4 .")
        
        w = np.zeros(nInput)
        w[0] = np.floor((N-1)/(2*M))
        max_wi = np.floor(w[0]/(2*M))
        
        if max_wi >= nInput-1:
            w[1:] = np.floor(np.linspace(1, max_wi, nInput-1))
        else:
            w[1:] = np.arange(nInput-1)%max_wi+1
        
        s = (2*np.pi/N)*np.arange(N)
        
        X = np.zeros((N*nInput, nInput))
        w_tmp = np.zeros(nInput)
        
        for i in range(nInput):
            w_tmp[i] = w[0]
            idx = list(range(i))+list(range(i+1,nInput))
            w_tmp[idx] = w[1:]
            idx = range(i*N, (i+1)*N)   
            phi = 2*np.pi*np.random.rand()    
            sin_result = np.sin(w_tmp[:,None]*s+phi)
            arsin_result = (1/np.pi)*np.arcsin(sin_result) #saltelli formula
            X[idx, :] = 0.5+arsin_result.transpose()
        
        return problem._unit_X_transform(X)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None):
        '''
            Perform FAST or extend FAST analysis
            Noted that if the X and Y is None, sample(500) is used for generate data 
                       and use the method problem.evaluate to evaluate them.
            -------------------------------------------------
            Parameters:
                X: np.ndarray
                    the input data
                Y: np.ndarray
                    the result data
            Returns:
                Si: dict
                    The type of Si is dict. And it contain 'S1', 'ST' key value. 
        '''
        #Parameter Setting
        M = self.getParaValue('M')
        
        #Set problem
        self.setProblem(problem)
        
        if Y is None:
            Y = self.evaluate(X)
        
        X, Y = self.__check_and_scale_xy__(X, Y)
        
        nInput = problem.nInput; 
        n = int(X.shape[0] / nInput)
        
        S1 = np.zeros(nInput); ST = np.zeros(nInput)
        
        #main process
        w_0 = np.floor((n-1)/(2*M))
             
        for i in range(nInput):
            idx = np.arange(i*n, (i+1)*n)
            Y_sub = Y[idx]
            f = np.fft.fft(Y_sub.ravel())
            Sp = np.power(np.absolute(f[np.arange(1, np.ceil(n / 2), dtype=np.int32)]) / n, 2)
            V = 2.0*np.sum(Sp)
            Di = 2.0*np.sum(Sp[np.int32(np.arange(1, M+1, dtype=np.int32)*w_0-1)]) #pw<=(NS-1)/2 w_0=(NS-1)/M
            Dt = 2.0*np.sum(Sp[np.arange(np.floor(w_0/2.0), dtype=np.int32)])
            
            S1[i] = Di/V
            ST[i] = 1.0-Dt/V
        
        self.record('S1', problem.xLabels, S1)
        self.record('ST', problem.xLabels, ST)
        
        return self.result
