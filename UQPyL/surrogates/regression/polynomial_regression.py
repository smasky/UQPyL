import numpy as np
from math import comb
from typing import Literal, Tuple, Optional

from .linear_regression import LinearRegression
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures

class PolynomialRegression(LinearRegression):
    
    """
    PolynomialRegression
    """
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                interactionOnly: bool=False, degree: int=2, 
                lossType: Literal['Origin', 'Ridge', 'Lasso']='Origin',
                fitIntercept: bool= True,
                C: float=0.1, C_ub: float=100, C_lb: float=1e-5):
        
        super().__init__(scalers=scalers, polyFeature=None,
                         lossType=lossType, fitIntercept=fitIntercept, C=C, C_ub=C_ub, C_lb=C_lb)
        
        self.degree=degree
        self.fitIntercept=fitIntercept
        self.interactionOnly=interactionOnly
        
###------------------------public functions-----------------------------###
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain=self.__check_and_scale__(xTrain, yTrain)
        
        xTrain=self.polynomialFeatures(xTrain)
        
        if self.lossType=='Origin':
            self.fitOrigin(xTrain, yTrain)
        elif self.lossType=='Ridge':
            self.fitRidge(xTrain, yTrain)
        elif self.lossType=='Lasso':
            self.fitLasso(xTrain, yTrain)
        else:
            raise ValueError('Using wrong model type!')
        
    def predict(self, xPred: np.ndarray) -> np.ndarray:
        
        xPred=self.__X_transform__(xPred)
        xPred=self.polynomialFeatures(xPred)
        
        if self.fit_intercept:
            yPred=xPred@self.coef+self.intercept
        else:
            yPred=xPred@self.coef
            
        yPred=yPred.reshape(-1,1)
        
        return self.__Y_inverse_transform__(yPred)
    
###------------------------private functions-----------------------------###
    def polynomialFeatures(self, xTrain: np.ndarray):
        
        nSample, nFeature=xTrain.shape
        n_output_features=0
        
        if self.interactionOnly:
            for d in range(1, self.degree+1):
                n_output_features+=comb(nFeature, d)
        else: 
            for d in range(1, self.degree+1):
                n_output_features+=comb(d+(nFeature-1), nFeature-1)
                
        outTrainX=np.zeros((nSample, n_output_features))
        current_col=0
             
        ######################degree1#####################
        
        outTrainX[:, current_col : current_col + nFeature] = xTrain
        index = list(range(current_col, current_col + nFeature))
        current_col += nFeature
        index.append(current_col)
        
        ####################degree>2####################
        
        for _ in range(2, self.degree + 1):
            new_index = []
            end = index[-1]
            for feature_idx in range(nFeature):
                start = index[feature_idx]
                new_index.append(current_col)
                if self.interaction_only:
                    start += index[feature_idx + 1] - index[feature_idx]
                next_col = current_col + end - start
                if next_col <= current_col:
                    break
                np.multiply(
                    outTrainX[:, start:end],
                    xTrain[:, feature_idx : feature_idx + 1],
                    out=outTrainX[:, current_col:next_col]
                )
                current_col = next_col
            new_index.append(current_col)
            index = new_index
            
        return outTrainX