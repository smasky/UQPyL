# Extend Fourier amplitude sensitivity test, FAST
"""
  References
    ----------
    1. Cukier, R.I., Fortuin, C.M., Shuler, K.E., Petschek, A.G.,
       Schaibly, J.H., 1973.
       Study of the sensitivity of coupled reaction systems to
       uncertainties in rate coefficients. I theory.
       Journal of Chemical Physics 59, 3873-3878.
       doi:10.1063/1.1680571
    2. Saltelli, A., S. Tarantola, and K. P.-S. Chan (1999).
       A Quantitative Model-Independent Method for Global Sensitivity Analysis
       of Model Output.
       Technometrics, 41(1):39-56,
       doi:10.1080/00401706.1999.10485594
    3. extend FAST
"""
import numpy as np
from typing import Optional, Tuple

from .sa_ABC import SA
from ..problems import ProblemABC as Problem
from ..utility import Scaler
class FAST(SA):
    def __init__(self, problem: Problem, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                       M: int=4):
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
    
        super().__init__(problem, scalers)

        self.M=M
        
    def sample(self, N: int=500) -> np.ndarray:
        '''
            Generate FAST sequence, this technique from paper [2]
            --------------------------
            Parameters:
                N: int, default=500
                    the number of sample points for each sequence

            Returns:
            X: 2d-np.ndarray
                the size of X is ((N*n_input, n_input))
        '''
        n_input=self.n_input; M=self.M
        
        if N<4*M**2:
            raise ValueError("the number of sample must be greater than 4*M**2!")
        
        w=np.zeros(n_input)
        w[0]=np.floor((N-1)/(2*M))
        max_wi=np.floor(w[0]/(2*self.M)) #Saltelli
        
        if max_wi>=n_input-1:
            w[1:]=np.floor(np.linspace(1, max_wi, n_input-1))
        else:
            w[1:]=np.arange(n_input-1)%max_wi+1
        
        s=(2*np.pi/N)*np.arange(N)
        
        X=np.zeros((N*n_input, n_input))
        w_tmp=np.zeros(n_input)
        
        for i in range(n_input):
            w_tmp[i]=w[0]
            idx=list(range(i))+list(range(i+1,n_input))
            w_tmp[idx]=w[1:]
            idx=range(i*N, (i+1)*N)   
            phi=2*np.pi*np.random.rand()    
            sin_result=np.sin(w_tmp[:,None]*s+phi)
            arsin_result=(1/np.pi)*np.arcsin(sin_result) #saltelli formula
            X[idx, :]=0.5+arsin_result.transpose()
        
        return self.transform_into_problem(X)
        
    def analyze(self, X: np.ndarray=None, Y: np.ndarray=None, verbose: bool=False) -> dict:
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
                verbose: bool
                    the switch to print analysis summary or not     
            Returns:
                Si: dict
                    The type of Si is dict. And it contain 'S1', 'ST' key value. 
        '''
        
        X, Y=self.__check_and_scale_xy__(X, Y)
        n_input=self.n_input; M=self.M;N=int(X.shape[0]/n_input)
        S1=np.zeros(n_input); ST=np.zeros(n_input);
        
        #main process
        w_0=np.floor((N-1)/(2*M))
             
        for i in range(n_input):
            idx=np.arange(i*N, (i+1)*N)
            Y_sub=Y[idx]
            #fft
            f=np.fft.fft(Y_sub.ravel())
            # Sp = np.power(np.absolute(f[np.arange(1, np.ceil((self.N_within_sampler-1)/2), dtype=np.int32)-1])/self.N_within_sampler, 2) #TODO 1-(NS-1)/2 
            Sp = np.power(np.absolute(f[np.arange(1, np.ceil(N / 2), dtype=np.int32)]) / N, 2)
            V=2.0*np.sum(Sp)
            Di=2.0*np.sum(Sp[np.int32(np.arange(1, M+1, dtype=np.int32)*w_0-1)]) #pw<=(NS-1)/2 w_0=(NS-1)/M
            Dt=2.0*np.sum(Sp[np.arange(np.floor(w_0/2.0), dtype=np.int32)])
            
            S1[i]=Di/V
            ST[i]=1.0-Dt/V
        
        Si={'S1':S1, 'ST':ST}
        self.Si=Si
        
        if verbose:
            self.summary()
        
        return Si
        
    def summary(self):
        '''
            print analysis summary
        '''
        if self.Si is None:
            raise ValueError("Please run analyze method first!")
        
        print("FAST Sensitivity Analysis")
        print("-------------------------------------------------")
        print("Input Dimension: %d" % self.n_input)
        print("-------------------------------------------------")
        print("First Order Sensitivity Indices: ")
        print("-------------------------------------------------")
        for label, value in zip(self.x_labels, self.Si['S1']):
            print(f"{label}: {value:.4f}")
        print("-------------------------------------------------")
        print("Total Order Sensitivity Indices: ")
        print("-------------------------------------------------")
        for i in range(self.n_input):
           print(f"{self.x_labels[i]}: {self.Si['ST'][i]:.4f}")
        print("-------------------------------------------------")
        print("-------------------------------------------------")
    
    #--------------------Private Function---------------------------#
    def _default_sample(self):
        return self.sample(500)