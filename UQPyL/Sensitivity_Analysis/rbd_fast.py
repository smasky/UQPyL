#Random Balance Designs Fourier Amplitude Sensitivity Test
import numpy as np
from ..Experiment_Design import LHS
from ..Utility import MinMaxScaler
from scipy.signal import periodogram
from scipy.stats import norm
lhs=LHS('center')
class RBD_FAST():
    def __init__(self, problem, NSample=1000, surrogate=None, M=10, NSurrogate=50, XInit=None, YInit=None):
        self.evaluate=problem.evaluate
        self.lb=problem.lb; self.ub=problem.ub
        self.dim=problem.dim
        self.surrogate=surrogate
        
        self.M=M
        self.NSample=NSample
        
        if self.NSample<=4*self.M**2:
            raise ValueError("the number of sample must be greater than 4*M**2!")
            
        self.NSample=NSample
        
        if self.surrogate:
            if XInit is None:
                self.XInit=lhs(NSurrogate, self.dim)*(self.ub-self.lb)+self.lb
    
    def analyze(self):
        Si=[]
        
        X_seq=lhs(self.NSample, self.dim)*(self.ub-self.lb)+self.lb
        if self.surrogate:
            self.YInit=self.evaluate(self.XInit)
            self.surrogate.fit(self.XInit, self.YInit)
            Y_seq=self.surrogate.predict(self.X_seq)
        else:
            Y_seq=self.evaluate(X_seq)
        scaler=MinMaxScaler(0,1)
        for i in range(self.dim):
            idx=np.argsort(X_seq[:, i])
            idx=np.concatenate([idx[::2], idx[1::2][::-1]])
            Y_tmp=scaler.fit_transform(Y_seq[idx])
            _,Pxx=periodogram(Y_tmp)
            V=np.sum(Pxx[1:])
            D1=np.sum(Pxx[1: self.M+1])
            S1=V/D1
            lamb=(2*self.M)/None
            S1=S1-lamb/(1-lamb)*(1-S1)
            
            Si.append(V/D1)
            
        return Si
            
            
            
        
        