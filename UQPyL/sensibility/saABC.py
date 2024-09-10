import abc
from typing import Tuple, Optional
import numpy as np

from ..DoE import LHS, Sampler
from ..utility import Scaler, MinMaxScaler
from ..surrogates import Surrogate
from ..problems import ProblemABC as Problem

class SA(metaclass=abc.ABCMeta):
    
    Si=None
    firstOrder=False; secondOrder=False; totalOrder=False
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]], 
                 verbose: bool=False):
             
        if scalers[0] is None:
            self.xScale=None
        else:
            if not isinstance(scalers[0], Scaler):
                    raise TypeError("scaler must be an instance of Scaler or None!")
            self.xScale=scalers[0]
        
        if scalers[1] is None:
            self.yScale=None
        else:
            if not isinstance(scalers[1], Scaler):
                    raise TypeError("scaler must be an instance of Scaler or None!")
            self.yScale=scalers[1]

        self.setting=Setting()
        self.verbose=verbose
        
    def setParameters(self, key, value):
        
        self.setting.setParameter(key, value)
    
    def getParaValue(self, *args):
        
        return self.setting.getParaValue(*args) 
        
    def setProblem(self, problem: Problem):
        
        self.problem=problem
        self.nInput=problem.nInput
        self.lb=problem.lb; self.ub=problem.ub
        self.labels=problem.x_labels
    
    def __check_and_scale_xy__(self, X, Y):
        
        if not isinstance(X, np.ndarray) and X is not None:
            raise TypeError("X must be an instance of np.ndarray or None!")
         
        if self.xScale:
            X=self.xScale.fit_transform(X)
        
        if not isinstance(Y, np.ndarray) and Y is not None:
            raise TypeError("Y must be an instance of np.ndarray or None!")

        if self.yScale:
            Y=self.yScale.fit_transform(Y)
                  
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
    
    def transform_into_problem(self, X, problem):
        
        return X*(problem.ub-problem.lb)+problem.lb
    
    @abc.abstractmethod
    def analyze(self, X_sa=None, Y_sa=None):
        pass
    

class Setting():
    """
    Save the parameter setting of the algorithm
    """
    
    def __init__(self):
        self.keys=[]
        self.values=[]
        self.dicts={}
    
    def setParameter(self, key, value):
        
        self.dicts[key]=value
        self.keys.append(key)
        self.values.append(value)
    
    def getParaValue(self, *args):
        
        values=[]
        for arg in args:
            values.append(self.dicts[arg])
        
        if len(args)>1:
            return tuple(values)
        else:
            return values[0]