import numpy as np
from typing import Optional, Tuple

from ..surrogates import MARS, Surrogate
from ..utility import MinMaxScaler, Scaler
from ..problems import Problem_ABC as Problem
from ..DoE import LHS, Sampler
from .sa_ABC import SA


class MARS_SA(SA):
    def __init__(self, problem: Problem, 
                 sampler: Sampler=LHS('classic'), N_within_sampler: int=100,
                 scale: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 surrogate: Surrogate=None, if_sampling_consistent: bool=False,
                 sampler_for_surrogate: Sampler=None, N_within_surrogate_sampler: int=50,
                 X_for_surrogate=None, Y_for_surrogate=None):
        
        super().__init__(problem, sampler, N_within_sampler,
                         scale, surrogate, if_sampling_consistent,
                         sampler_for_surrogate, N_within_surrogate_sampler,
                         X_for_surrogate, Y_for_surrogate
                         )
    
    def analyze(self, X_sa=None, Y_sa=None):
        
        ##forward process
        X_sa=self.__check_and_scale_x__(X_sa)
        self.__prepare_surrogate__()
        Y_sa=self.evaluate(X_sa)
        
        S1=[]
        #main process    
        mars=MARS(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)))
        mars.fit(X_sa, Y_sa)
        base_gcv=mars.gcv_
        
        for i in range(self.n_input):
            X_sub=np.delete(X_sa, [i], axis=1)
            mars=MARS(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)))
            mars.fit(X_sub, Y_sa)
            S1.append(np.abs(base_gcv-mars.gcv_))
            
        S1_sum = sum(S1)
        S1_normalized = [s/S1_sum for s in S1]
        return S1_normalized
    
    def summary(self):
        pass
        #TODO summary
        
        
        
        
        
        
        
        
        
        