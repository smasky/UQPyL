import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Tuple
from scipy.stats import cramervonmises_2samp

from ..optimization import Binary_GA
from .sa_ABC import SA
from ..DoE import LHS, FAST_Sampler, Sampler
from ..problems import Problem_ABC as Problem
from ..utility import Scaler
from ..surrogates import Surrogate

class RSA(SA):
    def __init__(self, problem: Problem, n_region: int=20,
                 sampler: Sampler=LHS('classic'), N_within_sampler=100,
                 scale: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), surrogate: Surrogate=None, if_sampling_consistent: bool=False,
                 sampler_for_surrogate: Sampler=None, N_within_surrogate_sampler: int=50,
                 X_for_surrogate: np.ndarray=None, Y_for_surrogate: np.ndarray=None):
        
        super().__init__(problem, sampler, N_within_sampler,
                         scale, surrogate, if_sampling_consistent,
                         sampler_for_surrogate, N_within_surrogate_sampler,
                         X_for_surrogate, Y_for_surrogate
                         )

        self.n_region=n_region
    
    def analyze(self, X_sa=None, Y_sa=None):
        
        ##forward process
        X_sa=self.__check_and_scale_x__(X_sa)
        self.__prepare_surrogate__()
        Y_sa=self.evaluate(X_sa)
        
        seq = np.linspace(0, 1, self.n_region + 1)
        results = np.full((self.n_region, self.n_input), np.nan)
        X_di = np.empty(X_sa.shape[0])
        
        for d_i in range(self.n_input):
                X_di = X_sa[:, d_i]
                for bin_index in range(self.n_region):
                    lower_bound, upper_bound = seq[bin_index], seq[bin_index + 1]
                    b = (lower_bound < X_di) & (X_di <= upper_bound)
                    if np.count_nonzero(b) > 0 and np.unique(X_sa[b]).size > 1:
                        r_s = cramervonmises_2samp(Y_sa[b].ravel(), Y_sa[~b].ravel()).statistic
                        results[bin_index, d_i] = r_s

        return results
        
    
    