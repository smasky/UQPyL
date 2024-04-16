#Delta test

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Tuple

from ..optimization import Binary_GA
from .sa_ABC import SA
from ..DoE import LHS, FAST_Sampler, Sampler
from ..problems import Problem_ABC as Problem
from ..utility import Scaler
from ..surrogates import Surrogate
class Delta_Test(SA):
    def __init__(self, problem: Problem, n_neighbors: int=2,
                 sampler: Sampler=LHS('classic'), N_within_sampler=100,
                 scale: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), surrogate: Surrogate=None, if_sampling_consistent: bool=False,
                 sampler_for_surrogate: Sampler=None, N_within_surrogate_sampler: int=50,
                 X_for_surrogate: np.ndarray=None, Y_for_surrogate: np.ndarray=None):
        
        super().__init__(problem, sampler, N_within_sampler,
                         scale, surrogate, if_sampling_consistent,
                         sampler_for_surrogate, N_within_surrogate_sampler,
                         X_for_surrogate, Y_for_surrogate
                         )

        self.n_neighbors=n_neighbors
         
    def analyze(self, X_sa=None, Y_sa=None):

        ##forward process
        X_sa=self.__check_and_scale_x__(X_sa)
        self.__prepare_surrogate__()
        Y_sa=self.evaluate(X_sa)
        
        ##main process
        self.X_sa=X_sa; self.Y_sa=Y_sa
        optimizer=Binary_GA(self.cal_delta, self.n_input)
        self.best_paras, self.best_value, history_paras, _=optimizer.run()
        
        S1_score=np.sum(history_paras, axis=0)/history_paras.shape[0]
        
        return S1_score
    
    def summary(self):
        
        if self.best_paras==None or self.best_value==None:
            raise ValueError("Please run analyze() first!")
        
        idx=[index for index, value in enumerate(self.best_value) if value==1]
        
        print('The best performance parameter combinations:')
        print(' |'.join(self.labels[idx]))
        print(' |'.join(map(str, self.S1_score[idx])))
         
        #TODO output the optimal variables

    def cal_delta(self, exclude_feature_list):
        
        #TODO expensive computation so using pybind11 or cython to accelerate 
        X=np.copy(self.X_sa)
        y=np.copy(self.Y_sa)
        exclude_feature = [index for index, value in enumerate(exclude_feature_list) if value == 0]
        
        if exclude_feature is not None:
            X = np.delete(X, exclude_feature, axis=1)
            
        distances = cdist(X, X)
        np.fill_diagonal(distances, np.inf)
        
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        deltas = []
        for i in range(len(X)):
            neighbor_deltas = (y[i] - y[neighbors_indices[i]])**2
            delta = np.mean(neighbor_deltas)
            deltas.append(delta)
        
        return np.mean(deltas)     