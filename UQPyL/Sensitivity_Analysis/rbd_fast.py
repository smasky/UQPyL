#Random Balance Designs Fourier Amplitude Sensitivity Test
import numpy as np
from .sa_ABC import SA

from scipy.signal import periodogram
from scipy.stats import norm

class RBD_FAST(SA):
    def __init__(self, problem, n_sample=1000, M=10,
                 scale=None, lhs=None,
                 surrogate=None, n_surrogate_sample=50,
                 X_for_surrogate=None, Y_for_surrogate=None):
        
        super().__init__(problem, n_sample,
                         scale, lhs,
                         surrogate, n_surrogate_sample, X_for_surrogate, Y_for_surrogate)
        
        self.M=M

        if self.n_sample<=4*self.M**2:
            raise ValueError("the number of sample must be greater than 4*M**2!")
                
    def analyze(self, X_sa=None, Y_sa=None):
        
        X_sa, Y_sa=self.__check_and_scale_x_y__(X_sa, Y_sa)
        
        Si=[]
        
        for i in range(self.dim):
            idx=np.argsort(X_sa[:, i])
            idx=np.concatenate([idx[::2], idx[1::2][::-1]])
            Y_tmp=Y_sa[idx]
           
            Pxx = np.abs(np.fft.rfft(Y_tmp[:, 0]))**2 / self.n_sample / (self.n_sample - 1)

            V=np.sum(Pxx[1:])
            D1=np.sum(Pxx[1: self.M+1])
            S1=V/D1
            lamb=(2*self.M)/self.n_sample
            S1=S1-lamb/(1-lamb)*(1-S1)
            
            Si.append(S1)
            
        return Si
    
    def summary(self):
        #TODO summary 
        pass
            
            
            
        
        