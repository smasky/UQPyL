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
    X_scaler=None
    Y_scaler=None
    train_x=None
    train_y=None

    def __init__(self, scalers=(None, None), poly_feature=None):
        
        if scalers[0]:
            self.X_scaler=scalers[0]
        else:
            self.X_scaler=None
        
        if scalers[1]:
            self.Y_scaler=scalers[1]
        else:
            self.Y_scaler=None
        
        if poly_feature:
            self.poly_feature=poly_feature
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
            
            
            if(self.X_scaler):
                train_x=self.X_scaler.fit_transform(train_x)
        
            if(self.Y_scaler):
                train_y=self.Y_scaler.fit_transform(train_y)

            if(self.poly_feature):
                train_x=self.poly_feature.transform(train_x)
                
            self.n_features=train_x.shape[1]
            
            return train_x,train_y
        else:
            raise ValueError("The shapes of x and y are not consistent. Please check them!")
    
    def __X_transform__(self,X: np.ndarray) -> np.ndarray:
        
        if (self.X_scaler):
            X=self.X_scaler.transform(X)
        
        if (self.poly_feature):
            X=self.poly_feature.transform(X)
            
        return X
    
    def __Y_transform__(self, Y: np.ndarray) -> np.ndarray:
        
        if(self.Y_scaler):
            Y=self.Y_scaler.transform(Y.reshape(-1,1))
            
        return Y
    
    def __Y_inverse_transform__(self, Y: np.ndarray) -> np.ndarray:

        if(self.Y_scaler):
            Y=self.Y_scaler.inverse_transform(Y.reshape(-1,1))
            
        return Y
    def __X_inverse_transform__(self, X: np.ndarray) -> np.ndarray:
        
        if(self.X_scaler):
            X=self.X_scaler.inverse_transform(X)
        
        return X
    
    def set_Paras(self, Para_dicts):
        
        for name, value in Para_dicts.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                raise ValueError("Cannot found this parameter! Please check")
            
    @abc.abstractmethod
    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        pass
    
    @abc.abstractmethod
    def predict(self, predict_X: np.ndarray):
        pass
    
class Mo_Surrogates():
    def __init__(self, n_surrogates, models_list=[]):
        from .surrogate_ABC import Surrogate
        self.n_surrogates=n_surrogates
        
        for model in models_list:
            if not isinstance(model, Surrogate):
                ValueError("Please append the type of surrogate!") 
                         
        self.models_list=models_list
        
    def append(self, model):
        from .surrogate_ABC import Surrogate
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