import numpy as np
from ..Experiment_Design import LHS
from ..Surrogates import MARS
from ..Utility import MinMaxScaler

lhs=LHS('center')
class MARS_SA():
    def __init__(self, problem, NSample=1000, 
                       surrogate=None, NSurrogate=50, XInit=None, YInit=None):
        self.evaluate=problem.evaluate
        self.dim=problem.dim
        self.ub=problem.ub; self.lb=problem.lb
        self.surrogate=surrogate
        
        self.NSample=NSample
        
        if self.surrogate:
            if XInit is None:
                self.XInit=lhs(NSurrogate, self.dim)*(self.ub-self.lb)+self.lb
    
    def analyze(self):
        
        
        X_seq=lhs(self.NSample, self.dim)*(self.ub-self.lb)+self.lb
        if self.surrogate:
            self.YInit=self.evaluate(self.XInit)
            self.surrogate.fit(self.XInit, self.YInit)
            Y_seq=self.surrogate.predict(X_seq)
        else:
            Y_seq=self.evaluate(X_seq)
            
        mars=MARS(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)))
        mars.fit(X_seq, Y_seq)
        base_gcv=mars.gcv_
        
        Si=[]
        
        for i in range(self.dim):
            X_sub=np.delete(X_seq, [i], axis=1)
            mars=MARS(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)))
            mars.fit(X_sub, Y_seq)
            Si.append(np.abs(base_gcv-mars.gcv_))

        Si_sum = sum(Si)
        Si_normalized = [s/Si_sum for s in Si]
        return Si_normalized
        
        
        
        
        
        
        
        
        
        