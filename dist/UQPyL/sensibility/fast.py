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
from ..DoE import FAST_Sampler, Sampler, LHS
from ..problems import Problem_ABC as Problem
from ..surrogates import Surrogate
from ..utility import Scaler
class FAST(SA):
    def __init__(self, problem: Problem, sampler: Sampler=FAST_Sampler(M=4), N_within_sampler: int=100,
                 scale: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), surrogate: Surrogate=None, if_sampling_consistent: bool=False,
                 sampler_for_surrogate: Sampler=LHS('classic'), N_within_surrogate_sampler: int=50,
                 X_for_surrogate: Optional[np.ndarray]=None, Y_for_surrogate: Optional[np.ndarray]=None):
        
        super().__init__(problem, sampler, N_within_sampler,
                         scale, surrogate, if_sampling_consistent,
                         sampler_for_surrogate, N_within_surrogate_sampler,
                         X_for_surrogate, Y_for_surrogate
                         )

        if not isinstance(sampler, FAST_Sampler):
            raise TypeError("FAST only support for the FAST_Sampler, please check!")
        
    def analyze(self, X_sa=None, Y_sa=None):
        
        ##forward process
        X_sa=self.__check_and_scale_x__(X_sa)
        self.__prepare_surrogate__()
        Y_sa=self.evaluate(X_sa)  
        S1=[]; ST=[]
        
        #main process
        w_0=np.floor((self.N_within_sampler-1)/(2*self.sampler.M))
                
        for i in range(self.n_input):
            idx=np.arange(i*self.N_within_sampler, (i+1)*self.N_within_sampler)
            Y_sub=Y_sa[idx]
            #fft
            f=np.fft.fft(Y_sub)
            Sp = np.power(np.absolute(f[np.arange(1, np.ceil((self.N_within_sampler-1)/2), dtype=np.int32)-1])/self.N_within_sampler, 2) #TODO 1-(NS-1)/2

            V=2.0*np.sum(Sp)
            Di=2.0*np.sum(Sp[np.int32(np.arange(1, self.sampler.M+1, dtype=np.int32)*w_0-1)]) #pw<=(NS-1)/2 w_0=(NS-1)/M
            Dt=2.0*np.sum(Sp[np.arange(np.floor(w_0/2.0), dtype=np.int32)])
            
            S1.append(Di/V)
            ST.append(1.0-Dt/V)
            
        return np.array(S1), np.array(ST)
    
    def summary(self):
        pass
        #TODO summary the result of FAST
        
        
        
        