import abc
from typing import Tuple, Optional
import numpy as np

from ..Experiment_Design import Sampling, LHS
from ..Utility import Scaler, MinMaxScaler
from ..Surrogates import Surrogate
from ..Problems import Problem

class SA(metaclass=abc.ABCMeta):
    def __init__(self, problem, sampler: Sampling=LHS('center'), N_within_sampler: int=10,
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 surrogate: Optional[Surrogate]=None, if_sampling_consistent: bool=False,
                 sampler_for_surrogate: Sampling=LHS('center'), N_within_surrogate_sampler: int=50, 
                 X_for_surrogate: Optional[np.ndarray]=None, Y_for_surrogate: Optional[np.ndarray]=None):
        
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of Problem!")
        
        self.evaluate=problem.evaluate;self.n_input=problem.n_input
        self.lb=problem.lb; self.ub=problem.ub
        
        if problem.x_labels:
            self.labels=problem.x_labels
        else:
            self.labels=['x[%d]'%i for i in range(1, self.n_input+1)]
        
        for scale in scalers:
            if not isinstance(scale, Scaler) and scale is not None:
                raise TypeError("scale must be an instance of Scaler or None!")
        
        if scalers[0]:
            self.X_scale=MinMaxScaler(0,1)
        else:
            self.X_scale=scale[0]
        
        if scalers[1]:
            self.Y_scale=MinMaxScaler(0,1)
        else:
            self.Y_scale=scale[1]
        
        if not isinstance(sampler, Sampling):
            raise TypeError("sampler must be an instance of Sampling or None!")
        self.sampler=sampler; self.N_within_sampler=N_within_sampler
         
        if not isinstance(surrogate, Surrogate) and surrogate is not None:
            raise TypeError("surrogate must be an instance of Surrogate or None!")
        
        self.if_sampling_consistent=if_sampling_consistent
        self.surrogate=surrogate
        self.N_within_surrogate_sampler=N_within_surrogate_sampler
        self.sampler_for_surrogate=sampler_for_surrogate
        
        if X_for_surrogate is not None:
            if not isinstance(X_for_surrogate, np.ndarray):
                raise TypeError("X_for_surrogate must be an instance of np.ndarray or None!")
        
        if Y_for_surrogate is not None:
            if not isinstance(Y_for_surrogate, np.ndarray):
                raise TypeError("Y_for_surrogate must be an instance of np.ndarray or None!")
        
    def __check_and_scale_x_y__(self, X, Y):
        
        if not isinstance(X, np.ndarray) and X is not None:
            raise TypeError("X must be an instance of np.ndarray or None!")
        
        if not isinstance(Y, np.ndarray) and Y is not None:
            raise TypeError("Y must be an instance of np.ndarray or None!")
        
        if X==None:
            X=self.sampler.generate_sample(self.N_within_sampler)*(self.ub-self.lb)+self.lb
        
        if self.surrogate:
            if self.if_sampling_consistent:
                self.X_for_surrogate=np.copy(X)
            if self.X_for_surrogate is None:
                self.X_for_surrogate=self.sampler_for_surrogate(self.N_within_surrogate_sampler)*(self.ub-self.lb)+self.lb
            self.Y_for_surrogate=self.evaluate(self.X_for_surrogate)
            self.surrogate.fit(self.X_for_surrogate, self.Y_for_surrogate)
        
        if Y is None:    
            if self.surrogate:
                Y=self.surrogate.predict(X)
            else:
                Y=self.evaluate(X)
                
        X=np.atleast_2d(X)
        Y=np.atleast_2d(Y)
        
        if X.shape[0]!=Y.shape[0]:
            raise ValueError("The number of samples in X and Y must be the same!")
        
        
        #TODO 
        if self.X_scale:
            X=self.X_scale.fit_transform(X)
        
        if self.Y_scale:
            Y=self.Y_scale.fit_transform(Y)
        
        return X, Y
    
    
    @abc.abstractmethod
    def anlyze(self, X_sa=None, Y_sa=None):
        pass
    
    