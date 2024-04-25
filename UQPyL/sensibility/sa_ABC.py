import abc
from typing import Tuple, Optional
import numpy as np

from ..DoE import LHS, Sampler
from ..utility import Scaler, MinMaxScaler
from ..surrogates import Surrogate
from ..problems import ProblemABC as Problem

class SA(metaclass=abc.ABCMeta):
    Si=None
    def __init__(self, problem:Problem, scalers: Tuple[Optional[Scaler], Optional[Scaler]]):
        
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of Problem!")
           
        self.evaluate_=problem.evaluate
        self.n_input=problem.n_input
        self.lb=problem.lb; self.ub=problem.ub
        self.x_labels=problem.x_labels
        
        if problem.x_labels:
            self.labels=problem.x_labels
        else:
            self.labels=['x[%d]'%i for i in range(1, self.n_input+1)]
        
        if scalers[0] is None:
            self.X_scale=None
        else:
            if not isinstance(scalers[0], Scaler):
                    raise TypeError("scaler must be an instance of Scaler or None!")
            self.X_scale=scalers[0]
        
        if scalers[1] is None:
            self.Y_scale=None
        else:
            if not isinstance(scalers[1], Scaler):
                    raise TypeError("scaler must be an instance of Scaler or None!")
            self.Y_scale=scalers[1]
        
    def __check_and_scale_xy__(self, X, Y):
        
        if not isinstance(X, np.ndarray) and X is not None:
            raise TypeError("X must be an instance of np.ndarray or None!")
        elif X is None:
            X=self._default_sample()
         
        X=np.atleast_2d(X)
            
        if self.X_scale:
            X=self.X_scale.fit_transform(X)
        
        if not isinstance(Y, np.ndarray) and Y is not None:
            raise TypeError("Y must be an instance of np.ndarray or None!")
        elif Y is None:
            Y=self.evaluate_(X)

        if self.Y_scale:
            Y=self.Y_scale.fit_transform(Y)
            
        Y=Y.reshape(-1,1)
        
        return X, Y
    
    
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
                self.Y_scale.fitted=True
                
        return Y
    
    def transform_into_problem(self, X):
        
        return X*(self.ub-self.lb)+self.lb
    
    @abc.abstractmethod
    def analyze(self, X_sa=None, Y_sa=None):
        pass
    
    