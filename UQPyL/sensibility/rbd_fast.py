#Random Balance Designs Fourier Amplitude Sensitivity Test
import numpy as np
from scipy.signal import periodogram
from scipy.stats import norm

from .sa_ABC import SA

class RBD_FAST(SA):
    def __init__(self, problem, M=10, N_within_sampler=100,
                 scale=None, sampler=None,
                 surrogate=None, if_sampling_consistent=False,
                 sampler_for_surrogate=None, N_within_surrogate_sampler=50,
                 X_for_surrogate=None, Y_for_surrogate=None):
        
        super().__init__(problem, N_within_sampler,
                         scale, sampler,
                         surrogate, if_sampling_consistent,
                         sampler_for_surrogate, N_within_surrogate_sampler,
                        X_for_surrogate, Y_for_surrogate
                         )
        
        self.M=M

        if self.N_within_sampler<=4*self.M**2:
            raise ValueError("the number of sample must be greater than 4*M**2!")
                
    def analyze(self, X_sa=None, Y_sa=None):
        
        ##forward process
        X_sa=self.__check_and_scale_x__(X_sa)
        self.__prepare_surrogate__()
        Y_sa=self.evaluate(X_sa)
        
        S1=[]
        
        for i in range(self.n_input):
            idx=np.argsort(X_sa[:, i])
            idx=np.concatenate([idx[::2], idx[1::2][::-1]])
            Y_tmp=Y_sa[idx]
           
            Pxx = np.abs(np.fft.rfft(Y_tmp[:, 0]))**2 / self.N_within_sampler / (self.N_within_sampler - 1)

            V=np.sum(Pxx[1:])
            D1=np.sum(Pxx[1: self.M+1])
            S1=V/D1
            lamb=(2*self.M)/self.N_within_sampler
            S1=S1-lamb/(1-lamb)*(1-S1)
            
            S1.append(S1)
            
        return S1
    
    def summary(self):
        #TODO summary 
        pass
            
            
            
        
        