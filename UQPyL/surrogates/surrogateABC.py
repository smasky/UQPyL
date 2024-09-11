import abc
import numpy as np
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ..utility.scalers import StandardScaler, MinMaxScaler
from ..utility.polynomial_features import PolynomialFeatures
from typing import Literal,Tuple

Scale_T=Tuple[Literal['StandardScaler','MinMaxScaler'],Literal['StandardScaler','MinMaxScaler']]

class Surrogate(metaclass=abc.ABCMeta):
    Scale_type=None 
    normalized=None
    dim=None
    n_samples=None
    n_features=None
    xScaler=None
    yScaler=None
    train_x=None
    train_y=None

    def __init__(self, scalers=(None, None), polyFeature=None):
        
        self.setting=Setting()
        
        if scalers[0]:
            self.xScaler=scalers[0]
        else:
            self.xScaler=None
        
        if scalers[1]:
            self.yScaler=scalers[1]
        else:
            self.yScaler=None
        
        if polyFeature:
            self.poly_feature=polyFeature
        else:
            self.poly_feature=None
    
    def __check_and_scale__(self,train_x: np.ndarray, train_y: np.ndarray):
        '''
            check the type of train_data
                and normalize the train_data if required 
        '''
        
        if(not isinstance(train_x,np.ndarray) or not isinstance(train_y, np.ndarray)):
            raise ValueError('Please make sure the type of train_data is np.ndarry')
        
        self._train_x_=train_x.copy()
        self._train_y_=train_y.copy()
        
        train_x=np.atleast_2d(self._train_x_)
        train_y=np.atleast_2d(self._train_y_).reshape(-1, 1)
        
        if(train_x.shape[0]==train_y.shape[0]):
            
            self.n_samples=train_x.shape[0] 
            
            
            if(self.xScaler):
                train_x=self.xScaler.fit_transform(train_x)
        
            if(self.yScaler):
                train_y=self.yScaler.fit_transform(train_y)

            if(self.poly_feature):
                train_x=self.poly_feature.transform(train_x)
                
            self.n_features=train_x.shape[1]
            
            return train_x,train_y
        else:
            raise ValueError("The shapes of x and y are not consistent. Please check them!")
    
    def __X_transform__(self,X: np.ndarray) -> np.ndarray:
        
        if (self.xScaler):
            X=self.xScaler.transform(X)
        
        if (self.poly_feature):
            X=self.poly_feature.transform(X)
            
        return X
    
    def __Y_transform__(self, Y: np.ndarray) -> np.ndarray:
        
        if(self.yScaler):
            Y=self.yScaler.transform(Y.reshape(-1,1))
            
        return Y
    
    def __Y_inverse_transform__(self, Y: np.ndarray) -> np.ndarray:

        if(self.yScaler):
            Y=self.yScaler.inverse_transform(Y.reshape(-1,1))
            
        return Y
    def __X_inverse_transform__(self, X: np.ndarray) -> np.ndarray:
        
        if(self.xScaler):
            X=self.xScaler.inverse_transform(X)
        
        return X
    
    def set_Paras(self, Para_dicts):
        
        for name, value in Para_dicts.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                raise ValueError("Cannot found this parameter! Please check")
    
    def setParameters(self, key, value, lb, ub):
        
        self.setting.setParameter(key, value, lb, ub)
    
    def getParaValue(self, *args):
        
        return self.setting.getParaValue(*args)
    
    def addSetting(self, setting):
        
        self.setting.addSetting(setting)
     
    @abc.abstractmethod
    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        pass
    
    @abc.abstractmethod
    def predict(self, predict_X: np.ndarray):
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

class Setting():
    
    def __init__(self):
        
        self.hyperParas={}
        self.paraUb={}
        self.paraLb={}
    
    def addSubSetting(self, setting):
        
        prefix=setting.name
        self.hyperParas[prefix]=setting.hyperParas
        self.paraLb[prefix]=setting.paraLb
        self.paraUb[prefix]=setting.paraUb
    
    def keys(self):
        
        keyLists=[]
        
        
        return self.hyperParas.keys()
    
    def values(self):
        
        return self.hyperParas.values()
    
    def setParameter(self, key, value, lb, ub):
        
        self.hyperParas[key]=value
        self.paraLb[key]=lb
        self.paraUb[key]=ub

    def getParaValue(self, *args):
        
        values=[]
        for arg in args:
            values.append(self.hyperParas[arg])
        
        if len(args)>1:
            return tuple(values)
        else:
            return values[0]