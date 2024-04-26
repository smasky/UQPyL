import numpy as np
from math import comb
from typing import Literal, Tuple, Optional

from .surrogate_ABC import Scale_T, Surrogate
from .linear_regression import LinearRegression
from ..utility.scalers import Scaler
from ..utility.polynomial_features import PolynomialFeatures

class PolynomialRegression(LinearRegression):
    """
    PolynomialRegression
    """
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 poly_feature: PolynomialFeatures=None,
                 loss_type: Literal['Origin', 'Ridge', 'Lasso']='Origin',
                 interaction_only: bool=False,
                 degree: int=2, fit_intercept: bool=True,
                 alpha: float=0.6, epoch: int=100, lr: float=1e-6, tl: float=1e-4):
        
        super().__init__(scalers=scalers, poly_feature=poly_feature,
                         loss_type=loss_type, fit_intercept=fit_intercept, alpha=alpha,
                         epoch=epoch, lr=lr, tl=tl)
        
        self.degree=degree
        self.interaction_only=interaction_only
        
        if self.poly_feature and self.poly_feature.include_bias and self.fit_intercept:
            raise ValueError("The setting ( include_bias of PolyFeature and fit_intercept ) can not be turned on together.")
        
###------------------------public functions-----------------------------###
    def fit(self, trainX: np.ndarray, trainY: np.ndarray):
        
        trainX, trainY=self.__check_and_scale__(trainX, trainY)
        
        trainX_=self.polynomial_features(trainX)
        
        if self.loss_type=='Origin':
            self._fit_Origin(trainX_, trainY)
        elif self.loss_type=='Ridge':
            self._fit_Ridge(trainX_, trainY)
        elif self.loss_type=='Lasso':
            self._fit_Lasso(trainX_, trainY)
        else:
            raise ValueError('Using wrong model type!')
        
    def predict(self, predict_X: np.ndarray) -> np.ndarray:
        
        predict_X=self.__X_transform__(predict_X)
        predict_X_=self.polynomial_features(predict_X)
        if self.fit_intercept:
            predict_Y=predict_X_@self.coef_+self.intercept_
        else:
            predict_Y=predict_X_@self.coef_
        predict_Y=predict_Y.reshape(-1,1)
        return self.__Y_inverse_transform__(predict_Y)
    
###------------------------private functions-----------------------------###
    def polynomial_features(self, trainX: np.ndarray):
        
        n_samples, n_features=trainX.shape
        n_output_features=0
        
        if self.interaction_only:
            for d in range(1, self.degree+1):
                n_output_features+=comb(n_features, d)
        else: 
            for d in range(1, self.degree+1):
                n_output_features+=comb(d+(n_features-1),n_features-1)
                
        outTrainX=np.zeros((n_samples, n_output_features))
        current_col=0
             
        ######################degree1#####################
        
        outTrainX[:, current_col : current_col + n_features] = trainX
        index = list(range(current_col, current_col + n_features))
        current_col += n_features
        index.append(current_col)
        
        ####################degree>2####################
        
        for _ in range(2, self.degree + 1):
            new_index = []
            end = index[-1]
            for feature_idx in range(n_features):
                start = index[feature_idx]
                new_index.append(current_col)
                if self.interaction_only:
                    start += index[feature_idx + 1] - index[feature_idx]
                next_col = current_col + end - start
                if next_col <= current_col:
                    break
                np.multiply(
                    outTrainX[:, start:end],
                    trainX[:, feature_idx : feature_idx + 1],
                    out=outTrainX[:, current_col:next_col]
                )
                current_col = next_col
            new_index.append(current_col)
            index = new_index
            
        return outTrainX
    ############################Attribute#####################
    @property
    def degree(self):
        return self.degree_
    
    @degree.setter
    def degree(self, value):
        self.degree_=value
    