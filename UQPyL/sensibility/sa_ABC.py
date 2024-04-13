import abc
from typing import Tuple, Optional
import numpy as np

from ..DoE import Sampling, LHS
from ..utility import Scaler, MinMaxScaler
from ..surrogates import Surrogate
from ..problems import Problem

class SA(metaclass=abc.ABCMeta):
    def __init__(self, problem, sampler: Sampling=LHS('center'), N_within_sampler: int=100,
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 surrogate: Optional[Surrogate]=None, if_sampling_consistent: bool=False,
                 sampler_for_surrogate: Sampling=LHS('center'), N_within_surrogate_sampler: int=50, 
                 X_for_surrogate: Optional[np.ndarray]=None, Y_for_surrogate: Optional[np.ndarray]=None):
        
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of Problem!")
        
        self.evaluate_=problem.evaluate
        self.n_input=problem.n_input
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
            
            self.X_for_surrogate=X_for_surrogate
        
        if Y_for_surrogate is not None:
            if not isinstance(Y_for_surrogate, np.ndarray):
                raise TypeError("Y_for_surrogate must be an instance of np.ndarray or None!")
            
            self.Y_for_surrogate=Y_for_surrogate
        
    def __check_and_scale_x__(self, X):
        
        if not isinstance(X, np.ndarray) and X is not None:
            raise TypeError("X must be an instance of np.ndarray or None!")
                
        if X==None:
            X=self.sampler.sample(self.N_within_sampler, self.n_input)*(self.ub-self.lb)+self.lb
            
        if self.X_scale:
            X=self.X_scale.fit_transform(X)
            #reset Y_scale
            self.Y_scale.fitted=False
            
        if self.if_sampling_consistent:
            self.X_for_surrogate=np.copy(X)
                
        return X
    
    def __prepare_surrogate__(self):
        
        if self.surrogate:
            if self.X_for_surrogate is None:
                self.X_for_surrogate=self.sampler_for_surrogate.sample(self.N_within_surrogate_sampler, self.n_input)*(self.ub-self.lb)+self.lb
            self.Y_for_surrogate=self.evaluate(self.X_for_surrogate, origin=True)
            self.surrogate.fit(self.X_for_surrogate, self.Y_for_surrogate)

    def evaluate(self, X, origin=False):
       
        if not origin and self.surrogate:
            Y=self.surrogate.predict(X)
        else:
            Y=self.evaluate_(X)
        
        if self.Y_scale:
            if self.Y_scale.fitted:
                Y=self.Y_scale.transform(Y)
            else:        
                Y=self.Y_scale.fit_transform(Y)
                
        return Y
            
    
    @abc.abstractmethod
    def anlyze(self, X_sa=None, Y_sa=None):
        pass
    
    