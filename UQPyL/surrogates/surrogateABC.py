import abc
import numpy as np
from typing import Literal, Tuple

from .setting import Setting
from ..utility.scalers import StandardScaler, MinMaxScaler
from ..utility.polynomial_features import PolynomialFeatures

Scale_T=Tuple[Literal['StandardScaler','MinMaxScaler'], Literal['StandardScaler','MinMaxScaler']]

class Surrogate(metaclass=abc.ABCMeta):
    xScaler=None
    yScaler=None
    xTrain=None
    yTrain=None
    def __init__(self, scalers=(None, None), polyFeature=None):
        
        self.setting=Setting()
        
        self.xScaler=scalers[0] if scalers[0] else None
        self.yScaler=scalers[1] if scalers[1] else None
        self.polyFeature=polyFeature if polyFeature else None
    
    def __check_and_scale__(self, xTrain: np.ndarray, yTrain: np.ndarray):
        '''
            check the type of train data
                and normalize the train data if required 
        '''
        
        if(not isinstance(xTrain,np.ndarray) or not isinstance(yTrain, np.ndarray)):
            raise ValueError('Please make sure the type of train_data is np.ndarry')
                
        xTrain=np.atleast_2d(xTrain)
        yTrain=np.atleast_2d(yTrain).reshape(-1, 1)
        
        if(xTrain.shape[0]==yTrain.shape[0]):
            
            xTrain=self.xScaler.fit_transform(xTrain) if self.xScaler else np.copy(xTrain)
            
            yTrain=self.yScaler.fit_transform(yTrain) if self.yScaler else np.copy(yTrain)
            
            xTrain=self.polyFeature.transform(xTrain) if self.polyFeature else np.copy(xTrain)
            
            return xTrain,yTrain
        
        else:
            
            raise ValueError("The shapes of x and y are not consistent. Please check them!")
    
    def __X_transform__(self,X: np.ndarray) -> np.ndarray:
        
        X=self.xScaler.transform(X) if self.xScaler else X
        
        X=self.polyFeature.transform(X) if self.polyFeature else X
            
        return X
    
    def __Y_transform__(self, Y: np.ndarray) -> np.ndarray:
        
        Y=self.yScaler.transform(Y.reshape(-1,1)) if self.yScaler else Y
            
        return Y
    
    def __Y_inverse_transform__(self, Y: np.ndarray) -> np.ndarray:

        
        Y=self.yScaler.inverse_transform(Y.reshape(-1,1)) if self.yScaler else Y
            
        return Y
    
    def __X_inverse_transform__(self, X: np.ndarray) -> np.ndarray:
                
        X=self.xScaler.inverse_transform(X) if self.xScaler else X
        
        return X
    
    def setPara(self, key, value, lb, ub):
        
        self.setting.setPara(key, value, lb, ub)
    
    def getPara(self, *args):
        
        return self.setting.getPara(*args)
    
    def assignPara(self, key, value):
        
        self.setting.assignValues(key, value)
    
    def addSetting(self, setting):
        
        self.setting.addSubSetting(setting)
     
    @abc.abstractmethod
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        pass
    
    @abc.abstractmethod
    def predict(self, xPred: np.ndarray):
        pass
    
class Mo_Surrogates():
    def __init__(self, n_surrogates, models_list=[]):
        from .surrogateABC import Surrogate
        self.n_surrogates=n_surrogates
        
        for model in models_list:
            if not isinstance(model, Surrogate):
                ValueError("Please append the type of surrogate!") 
                         
        self.models_list=models_list
        
    def append(self, model):
        from .surrogateABC import Surrogate
        if not isinstance(model, Surrogate):
            ValueError("Please append the type of surrogate!")
            
        self.models_list.append(model)
    
    def fit(self, trainX: np.ndarray, trainY: np.ndarray):
        
        for i, model in enumerate(self.models_list):
            model.fit(trainX, trainY[:, i])
    
    def predict(self, testX: np.ndarray) -> np.ndarray:
        
        res=[]
        
        for model in self.models_list:
            res.append(model.predict(testX))
            
        pre_Y=np.hstack(res)
        
        return pre_Y