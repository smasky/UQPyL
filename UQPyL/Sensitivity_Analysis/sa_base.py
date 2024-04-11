import abc
from typing import Tuple, Optional
import numpy as np

from ..Experiment_Design import Sampling, LHS
from ..Utility import Scaler, MinMaxScaler
from ..Surrogates import Surrogate
from ..Problems import Problem

class SA_base(metaclass=abc.ABCMeta):
    def __init__(self, problem, n_sample: int =1000, 
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), 
                 lhs: Optional[Sampling]=None,
                 surrogate: Optional[Surrogate]=None, n_surrogate_sample: int=50, 
                 X_for_surrogate: Optional[np.ndarray]=None, Y_for_surrogate: Optional[np.ndarray]=None):
        
        self.n_sample=n_sample
        
        assert isinstance(problem, Problem), "problem must be an instance of Problem!"
        self.evaluate=problem.evaluate;self.n_input=problem.n_input
        self.lb=problem.lb; self.ub=problem.ub
        
        for scale in scalers:
            assert isinstance(scale, Scaler) or scale is None, "scale must be an instance of Scaler or None!"
        
        if scalers[0]:
            self.X_scale=MinMaxScaler(0,1)
        else:
            self.X_scale=scale[0]
        
        if scalers[1]:
            self.Y_scale=MinMaxScaler(0,1)
        else:
            self.Y_scale=scale[1]
        
        assert isinstance(lhs, Sampling) or lhs is None, "lhs must be an instance of Sampling or None!"
        if lhs:
            self.lhs=LHS('center')
        else:
            self.lhs=lhs
        
        assert isinstance(surrogate, Surrogate) or surrogate is None, "surrogate must be an instance of Surrogate or None!"
        self.surrogate=surrogate
        self.n_surrogate_sample=n_surrogate_sample
        
        if self.surrogate:
            if X_for_surrogate is None:
                self.X_for_surrogate=lhs(X_for_surrogate, self.dim)*(self.ub-self.lb)+self.lb
                self.Y_for_surrogate=Y_for_surrogate
    
    def __check_and_scale_x_y__(self, X, Y):
        
        assert isinstance(X, np.ndarray) or X is None, "X must be an instance of np.ndarray or None!"
        assert isinstance(Y, np.ndarray) or Y is None, "Y must be an instance of np.ndarray or None!"
                
        if X==None:
            X=self.lhs(self.n_sample, self.n_input)*(self.ub-self.lb)+self.lb
        
        if self.surrogate:
            if Y is None:
                if self.Y_for_surrogate is None:
                    self.Y_for_surrogate=self.evaluate(self.X_for_surrogate)
                Y=self.surrogate.predict(X)
            else:
                Y=self.evaluate(X)
                
        X=np.atleast_2d(X)
        Y=np.atleast_2d(Y)
        
        assert X.shape[0]==Y.shape[0], "The number of samples in X and Y must be the same!"
        
        if self.X_scale:
            X=self.X_scale.fit_transform(X)
        
        if self.Y_scale:
            Y=self.Y_scale.fit_transform(Y)
        
        return X, Y
    
    
    @abc.abstractmethod
    def anlyze(self, X_sa=None, Y_sa=None):
        pass
    
    