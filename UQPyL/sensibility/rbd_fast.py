#Random Balance Designs Fourier Amplitude Sensitivity Test
import numpy as np
from scipy.signal import periodogram
from scipy.stats import norm
from typing import Optional, Tuple

from .sa_ABC import SA
from ..DoE import Sampler, LHS
from ..problems import ProblemABC as Problem
from ..surrogates import Surrogate
from ..utility import Scaler
class RBD_FAST(SA):
    def __init__(self, problem: Problem, M: int=10, sampler: Sampler=LHS('classic'), N_within_sampler=1000, 
                 scale: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), surrogate: Surrogate=None,
                 if_sampling_consistent=False, sampler_for_surrogate: Sampler=LHS('classic'), N_within_surrogate_sampler: int=50,
                 X_for_surrogate: Optional[np.ndarray]=None, Y_for_surrogate: Optional[np.ndarray]=None):
        
        super().__init__(problem, sampler, N_within_sampler,
                         scale, surrogate, if_sampling_consistent,
                         sampler_for_surrogate, N_within_surrogate_sampler,
                         X_for_surrogate, Y_for_surrogate)
        self.M=M
        if N_within_sampler<=4*self.M**2:
            raise ValueError("the number of sample must be greater than 4*M**2!")
        
    def analyze(self, X_sa=None, Y_sa=None, verbose=False):
        
        ##forward process
        X_sa=self.__check_and_scale_x__(X_sa)
        self.__prepare_surrogate__()
        Y_sa=self.evaluate(X_sa)
        
        S1=[]
        
        self.X=X_sa; self.Y=Y_sa
        for i in range(self.n_input):
            idx=np.argsort(X_sa[:, i])
            idx=np.concatenate([idx[::2], idx[1::2][::-1]])
            Y_tmp=Y_sa[idx]
            
            # Pxx = np.abs(np.fft.rfft(Y_tmp[:, 0]))**2 / self.N_within_sampler / (self.N_within_sampler - 1)
            _, Pxx = periodogram(Y_tmp.ravel())
            V=np.sum(Pxx[1:])
            D1=np.sum(Pxx[1: self.M+1])
            S1_sub=D1/V
            
            #####normalization
            lamb=(2*self.M)/Y_sa.shape[0]
            S1_sub=S1_sub-lamb/(1-lamb)*(1-S1_sub)
            #####
            
            S1.append(S1_sub)
        
        Si={'S1':np.array(S1).ravel(), 'S2':None, 'ST':None}
        self.Si=Si
        
        if verbose:
            self.summary()
            
        return Si
    
    def summary(self):
        
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
            
        
        