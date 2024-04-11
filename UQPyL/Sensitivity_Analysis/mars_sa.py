import numpy as np


from ..Surrogates import MARS
from ..Utility import MinMaxScaler
from .sa_ABC import SA

class MARS_SA(SA):
    def __init__(self, problem, n_sample=100,
                 scale=None, lhs=None, 
                 surrogate=None, n_surrogate_sample=50, 
                 X_for_surrogate=None, Y_for_surrogate=None):
        
        super().__init__(problem, n_sample,
                         scale, lhs,
                         surrogate, n_surrogate_sample, X_for_surrogate, Y_for_surrogate
                         )
    
    def analyze(self, X_sa=None, Y_sa=None):
        
        X_sa, Y_sa=self.__check_and_scale_x_y__(X_sa, Y_sa)
            
        mars=MARS(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)))
        mars.fit(X_sa, Y_sa)
        base_gcv=mars.gcv_
        
        Si=[]
        
        for i in range(self.n_input):
            X_sub=np.delete(X_sa, [i], axis=1)
            mars=MARS(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)))
            mars.fit(X_sub, Y_sa)
            Si.append(np.abs(base_gcv-mars.gcv_))
            
        Si_sum = sum(Si)
        Si_normalized = [s/Si_sum for s in Si]
        return Si_normalized
    
    def summary(self):
        pass
        #TODO summary
        
        
        
        
        
        
        
        
        
        