import numpy as np
import abc


class Scaler(metaclass=abc.ABCMeta):
    def __init__(self):
        self.fitted=False
    
    @abc.abstractmethod
    def fit(self,trainX):
        self.fitted=True
    
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
    def __init__(self, min: int=0, max: int=1):

        self.min_scale=min
        self.max_scale=max
             
    def fit(self,trainX: np.ndarray):
        
        trainX=np.atleast_2d(trainX)
        self.min=np.min(trainX,axis=0)
        self.max=np.max(trainX,axis=0)
        super().fit(trainX)
        
    def transform(self, trainX: np.ndarray):
        
        trainX=np.atleast_2d(trainX)
        
        return (trainX-self.min)/(self.max-self.min)*(self.max_scale-self.min_scale)+self.min_scale
    
    def inverse_transform(self, trainX: np.ndarray):
        
        trainX=np.atleast_2d(trainX)
        
        return (trainX-self.min_scale)*(self.max-self.min)/(self.max_scale-self.min_scale)+self.min
    
    def fit_transform(self, trainX: np.ndarray):
        
        self.fit(trainX)
        
        return self.transform(trainX)

class StandardScaler(Scaler):
    def __init__(self, mu_x: int=0, sita_x: int=1):
        
        self.mu_x=mu_x
        self.sita_x=sita_x
    
    def fit(self,trainX: np.ndarray):
        
        trainX=np.atleast_2d(trainX)
        self.mu=np.mean(trainX, axis=0)
        self.sita=np.std(trainX, axis=0)
        
        super().fit(trainX)
        
    def transform(self, trainX: np.ndarray):
        
        trainX=np.atleast_2d(trainX)
        
        return (trainX-self.mu)/self.sita*self.sita_x+self.mu_x
    
    def inverse_transform(self, trainX: np.ndarray):
        
        trainX=np.atleast_2d(trainX)
        
        return ((trainX-self.mu_x)/self.sita_x)*self.sita+self.mu
    
    def fit_transform(self, trainX: np.ndarray):
        
        self.fit(trainX)
        
        return self.transform(trainX)
    