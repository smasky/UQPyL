import abc
from typing import Tuple, Optional
import numpy as np

from ..DoE import LHS, Sampler
from ..utility import Scaler, MinMaxScaler
from ..surrogates import Surrogate
from ..problems import ProblemABC as Problem

class SA(metaclass=abc.ABCMeta):
    
    result={}
    firstOrder=False; secondOrder=False; totalOrder=False
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]], 
                 verbose: bool=False, logFlag: bool=False, saveFlag: bool=False):
             
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

        self.verbose=verbose
        self.logFlag=logFlag
        self.saveFlag=saveFlag
        
        self.setting=Setting()
        self.result=Result(self)
        
        
    def setParameters(self, key, value):
        
        self.setting.setParameter(key, value)
    
    def getParaValue(self, *args):
        
        return self.setting.getParaValue(*args) 
        
    def setProblem(self, problem: Problem):
        
        self.problem=problem
    
    def record(self, key, x_labels,value):
        
        if not isinstance(value, np.ndarray):
            value=np.array(value)
        self.result.Si[key]=(x_labels, value)
        
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
    
    def transform_into_problem(self, problem, X):
        
        return X*(problem.ub-problem.lb)+problem.lb
    
    @abc.abstractmethod
    def analyze(self, X_sa=None, Y_sa=None):
        pass

class Result():
    
    def __init__(self, obj):
        
        self.Si={}
        self.labels=[]
        self.firstOrder=obj.firstOrder
        self.secondOrder=obj.secondOrder
        self.totalOrder=obj.totalOrder
        self.sa=obj
        
    def generateHDF5(self):
        
        x_labels=self.sa.problem.x_labels
        result={}
        
        for key, value in self.Si.items():
            x_labels=value[0]; matrix=value[1]
            result.setdefault(key, {})
            result[key]['matrix']=matrix
            for label, v in zip(x_labels, matrix.ravel()):
                result[key][label]=v
                    
        return result
class Setting():
    """
    Save the parameter setting of the algorithm
    """
    
    def __init__(self):
        self.dict={}
    
    def keys(self):
        return self.dict.keys()
    
    def values(self):
        
        return self.dict.values()
    
    def setParameter(self, key, value):
        
        self.dict[key]=value
    
    def getParaValue(self, *args):
        
        values=[]
        for arg in args:
            values.append(self.dict[arg])
        
        if len(args)>1:
            return tuple(values)
        else:
            return values[0]