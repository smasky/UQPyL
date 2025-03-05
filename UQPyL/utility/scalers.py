import numpy as np
import abc


class Scaler(metaclass = abc.ABCMeta):
    def __init__(self):
        self.fitted = False
    
    @abc.abstractmethod
    def fit(self,trainX):
        self.fitted = True
    
    @abc.abstractmethod
    def transform(self,trainX):
        pass
    
    @abc.abstractmethod
    def fit_transform(self,trainX):
        pass
    
    @abc.abstractmethod
    def inverse_transform(self,trainX):
        pass

class MinMaxScaler(Scaler):
    def __init__(self, min_: int = 0, max_: int = 1):

        self.min_scale = min_
        self.max_scale = max_
             
    def fit(self,trainX: np.ndarray):
        
        trainX = np.atleast_2d(trainX)
        self.min_ = np.min(trainX, axis=0)
        self.max_ = np.max(trainX, axis=0)
        super().fit(trainX)
        
    def transform(self, trainX: np.ndarray):
        
        trainX = np.atleast_2d(trainX)
        
        return (trainX - self.min_)/(self.max_ - self.min_) * (self.max_scale - self.min_scale) + self.min_scale
    
    def inverse_transform(self, trainX: np.ndarray):
        
        trainX = np.atleast_2d(trainX)
        
        return (trainX - self.min_scale)*(self.max_ - self.min_) / (self.max_scale - self.min_scale) + self.min_
    
    def fit_transform(self, trainX: np.ndarray):
        
        self.fit(trainX)
        
        return self.transform(trainX)

class StandardScaler(Scaler):
    def __init__(self, muX: int = 0, sitaX: int = 1):
        
        self.muX = muX
        self.sitaX = sitaX
    
    def fit(self,trainX: np.ndarray):
        
        trainX = np.atleast_2d(trainX)
        self.mu = np.mean(trainX, axis=0)
        self.sita = np.std(trainX, axis=0, ddof=1)
        
        super().fit(trainX)
        
    def transform(self, trainX: np.ndarray):
        
        trainX = np.atleast_2d(trainX)
        
        return (trainX-self.mu)/self.sita*self.sitaX+self.muX
    
    def inverse_transform(self, trainX: np.ndarray):
        
        trainX = np.atleast_2d(trainX)
        
        return ((trainX-self.muX)/self.sitaX)*self.sita+self.mu
    
    def fit_transform(self, trainX: np.ndarray):
        
        self.fit(trainX)
        
        return self.transform(trainX)
    