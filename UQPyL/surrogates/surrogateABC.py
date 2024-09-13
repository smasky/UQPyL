import abc
import numpy as np
from typing import Literal, Tuple
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
    
    def __check_and_scale__(self, xTrain: np.ndarray, yTrain: np.ndarray):
        '''
            check the type of train_data
                and normalize the train_data if required 
        '''
        
        if(not isinstance(xTrain,np.ndarray) or not isinstance(yTrain, np.ndarray)):
            raise ValueError('Please make sure the type of train_data is np.ndarry')
                
        xTrain=np.atleast_2d(xTrain)
        yTrain=np.atleast_2d(yTrain).reshape(-1, 1)
        
        if(xTrain.shape[0]==yTrain.shape[0]):
            
            if(self.xScaler):
                xTrain=self.xScaler.fit_transform(xTrain)
        
            if(self.yScaler):
                yTrain=self.yScaler.fit_transform(yTrain)

            if(self.poly_feature):
                xTrain=self.poly_feature.transform(xTrain)
                
            self.n_features=xTrain.shape[1]
            
            return xTrain,yTrain
        else:
            raise ValueError("The shapes of x and y are not consistent. Please check them!")
    
    def __X_transform__(self,X: np.ndarray) -> np.ndarray:
        
        if (self.xScaler):
            X=self.xScaler.transform(X)
        
        if (self.poly_feature):
            X=self.polyFeature.transform(X)
            
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
    
    def setPara(self, key, value, lb, ub):
        
        self.setting.setPara(key, value, lb, ub)
    
    def getPara(self, *args):
        
        return self.setting.getPara(*args)
    
    def addSetting(self, setting):
        
        self.setting.addSubSetting(setting)
     
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
        
        self.paras={}
        self.paras_ub={}
        self.paras_lb={}
    
    def getParaInfos(self):
        
        keyList=[]
        valueList=[]
        ubList=[]
        lbList=[]
        for key, item in self.paras.items():
            
            if isinstance(item, dict):
                keyList+=[f"{key}.{t}" for t in item.keys()]
                valueList+=item.values()
                ubList+=self.paras_ub[key].values()
                lbList+=self.paras_lb[key].values()
            else:
                keyList.append(key)
                valueList.append(item)
                ubList.append(self.paras_ub[key])
                lbList.append(self.paras_lb[key])
                
        return keyList, valueList, ubList, lbList
    
    def addSubSetting(self, setting):
        
        prefix=setting.prefix
        self.paras[prefix]=setting.paras
        self.paras_lb[prefix]=setting.paras_lb
        self.paras_ub[prefix]=setting.paras_ub
        
    def assignValues(self, keys, values):
        
        for i, key in enumerate(keys):
            lists=key.split('.')
            if len(lists)==1:
                self.paras[lists[0]]=values[i]
            else:
                self.paras[lists[0]][lists[1]]=values[i]

    def setPara(self, key, value, lb, ub):
        
        self.paras[key]=value
        self.paras_lb[key]=lb
        self.paras_ub[key]=ub
        
    def getPara(self, *args):
        
        values=[]
        for arg in args:
            values.append(self.paras[arg])
        
        if len(args)>1:
            return tuple(values)
        else:
            return values[0]