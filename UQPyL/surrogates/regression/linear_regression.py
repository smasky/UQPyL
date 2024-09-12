import numpy as np
from scipy.linalg import lstsq, solve
from typing import Tuple, Literal, Optional

from ..surrogateABC import Surrogate, Scale_T
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures

class LinearRegression(Surrogate):
    '''
    LinearRegression
    
    Support three version:
    'Origin'-------'Least Square Method'----Ordinary Loss Function
    'Ridge'--------'Ridge'----Using L2 regularization
    'Lasso'--------'Lasso'----Using L1 regularization
    '''
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 polyFeature: PolynomialFeatures=None, 
                 lossType: Literal['Origin', 'Ridge', 'Lasso']='Origin',
                 fitIntercept: bool= True, 
                 epoch: Optional[int]=None, tol: Optional[float]=None, 
                 C: float=0.1, C_ub: float=100, C_lb: float=1e-5):
        
        super().__init__(scalers, polyFeature)
        
        self.lossType=lossType
        self.fitIntercept=fitIntercept
        
        if lossType=="Lasso" or lossType=="Ridge":
            self.setPara("C", C, C_lb, C_ub)
            self.epoch=epoch
            self.tol=tol
                
        
###---------------------------------public function---------------------------------------###

    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain=self.__check_and_scale__(xTrain, yTrain)
        
        if self.loss_type=='Origin':
            
            self.fitOrigin(xTrain, yTrain)
            
        elif self.loss_type=='Ridge':
            
            self.fitRidge(xTrain, yTrain)
            
        elif self.loss_type=='Lasso':
            
            self.fitLasso(xTrain, yTrain)
            
        else:
            raise ValueError('Using wrong model type!')
        
    def predict(self, predict_X: np.ndarray) -> np.ndarray:
        
        predict_X=self.__X_transform__(predict_X)
        
        if(self.fit_intercept):
            predict_Y=predict_X@self.coef_+self.intercept_
        else:
            predict_Y=predict_X@self.coef_
        predict_Y=predict_Y.reshape(-1,1)
        return self.__Y_inverse_transform__(predict_Y)
    
###--------------------------private functions----------------------------###

    def fitOrigin(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        if self.fit_intercept:
            xTrain=np.hstack((xTrain, np.ones((xTrain.shape[0], 1))))
        
        self.coef, _ , self.rank, self.singular = lstsq(xTrain, yTrain)
        
        if self.fit_intercept:
            self.intercept = self.coef[-1]
            self.coef = self.coef[:-1]
        else:
            self.coef = self.coef
        
    def fitRidge(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        C = self.getPara("C")
        
        _, nFeatures=xTrain.shape
        
        if self.fit_intercept:
            xOffset = np.mean(xTrain, axis=0)
            yOffset = np.mean(yTrain, axis=0)
            xTrain -= xOffset
            yTrain -= yOffset
            
        xTrain.flat[::nFeatures+1]+=C
        A=np.dot(xTrain.T, xTrain)
        b=np.dot(xTrain.T, yTrain)
        
        self.coef=solve(A, b)
        
        if self.fit_intercept:
            self.intercept=yOffset-np.dot(xOffset.reshape(1,-1), self.coef_)
            return self.coef, self.intercept
        else:
            return self.coef
    
    def fitLasso(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        from .lasso import celer, compute_norms_X_col, compute_Xw, dnorm_enet
        
        l1_ratio=1.0
        
        C=self.getPara("C")
        
        xTrain=np.asarray(xTrain, order='F')
        yTrain=np.asarray(yTrain, order='F')
        nSamples, nFeatures=xTrain.shape
        
        xDense = xTrain
        xData = np.empty([1], dtype=xTrain.dtype)
        xIndices = np.empty([1], dtype=np.int32)
        xIndptr = np.empty([1], dtype=np.int32)
        
        if self.fit_intercept:
            xOffset=np.mean(xTrain, axis=0)
            yOffset=np.mean(yTrain, axis=0)
            xTrain-=xOffset
            yTrain-=yOffset
            
            xSparseScaling=xOffset
        else:
            xSparseScaling = np.zeros(nFeatures, dtype=xTrain.dtype)
        
        norms_X_col=np.zeros(nFeatures, dtype=xDense.dtype)
        compute_norms_X_col(
            False, norms_X_col, nSamples, xDense, xData,
            xIndices, xIndptr, xSparseScaling)
        
        w=np.zeros(nFeatures, dtype=xDense.dtype)
        Xw=np.zeros(nSamples, dtype=xDense.dtype)
        compute_Xw(False, 0, Xw, w, yTrain.ravel(), xSparseScaling.any(), xDense,
                    xData, xIndices, xIndptr, xSparseScaling)
        theta=Xw.copy()
        
        weights=np.ones(nFeatures, dtype=xDense.dtype)
        positive=False
       
        skip = np.zeros(xTrain.shape[1], dtype=np.int32)
        dnorm = dnorm_enet(False, theta, w, xDense, xData, 
                           xIndices, xIndptr, skip, xSparseScaling, 
                           weights, xSparseScaling.any(), positive,
                           C, l1_ratio)
        
        theta /= max(dnorm / (C * l1_ratio), nSamples)
        #
        maxIters=100 if self.epoch is None else self.epoch
        tl=1e-3 if self.tol is None else self.tol
        maxEpochs=500000; p0=10
        verbose=0; prune=True
        #
        sol = celer(False, 0, xDense, xData, xIndices, 
                    xIndptr, xSparseScaling, yTrain.ravel(),
                    C, l1_ratio, w, Xw, 
                    theta, norms_X_col, weights,
                    max_iter=maxIters, max_epochs=maxEpochs,
                    p0=p0, verbose=verbose, use_accel=1, tol=tl, prune=prune,
                    positive=positive)
        
        self.coef=sol[0]
        
        if self.fit_intercept:
            self.intercept=yOffset-np.dot(xOffset.reshape(1,-1), self.coef)
            return self.coef, self.intercept
        else:
            return self.coef
        
            
        
        
        
        

