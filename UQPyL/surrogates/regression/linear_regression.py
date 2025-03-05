import numpy as np
from scipy.linalg import lstsq, solve
from typing import Tuple, Literal, Optional, Union

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
    
    name = "LR"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                    polyFeature: PolynomialFeatures = None, 
                        lossType: Literal['Origin', 'Ridge', 'Lasso'] = 'Origin',
                            fitIntercept: bool = True,
                                C: float=0.1, 
                                C_attr: Union[dict, None] = {'ub': 100, 'lb': 1e-5, 'type': 'float', 'log': True},
                                maxIter: int = 100, maxEpoch: int = 5e5, tolerance: float = 1e-3, p0: int = 10
                                ):

        super().__init__(scalers, polyFeature)
        
        self.lossType = lossType
        self.fitIntercept = fitIntercept
        
        if lossType in ["Lasso", "Ridge"]:
            self.setting.setPara("C", C, C_attr)
            
            if lossType == "Lasso":
                self.setting.setPara("maxIter", maxIter)
                self.setting.setPara("maxEpoch", maxEpoch)
                self.setting.setPara("tol", tolerance)
                self.setting.setPara("p0", p0)
                
###---------------------------------public function---------------------------------------###

    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain = self.__check_and_scale__(xTrain, yTrain)
        
        if self.lossType == 'Origin':
            
            self.fitOrigin(xTrain, yTrain)
            
        elif self.lossType == 'Ridge':
            
            self.fitRidge(xTrain, yTrain)
            
        elif self.lossType == 'Lasso':
            
            self.fitLasso(xTrain, yTrain)
            
        else:
            raise ValueError('Using wrong model type!')
        
    def predict(self, xPred: np.ndarray) -> np.ndarray:
        
        xPred = self.__X_transform__(xPred)
        
        if(self.fitIntercept):
            yPred = xPred@self.coef+self.intercept
        else:
            yPred = xPred@self.coef
        yPred = yPred.reshape(-1,1)
        
        return self.__Y_inverse_transform__(yPred)
    
###--------------------------private functions----------------------------###
    def _fitPure(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        if self.lossType == 'Origin':
            
            self.fitOrigin(xTrain, yTrain)
            
        elif self.lossType == 'Ridge':
            
            self.fitRidge(xTrain, yTrain)
            
        elif self.lossType == 'Lasso':
            
            self.fitLasso(xTrain, yTrain)
            
        else:
            raise ValueError('Using wrong model type!')
        
    def fitOrigin(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        if self.fitIntercept:
            xTrain = np.hstack((xTrain, np.ones((xTrain.shape[0], 1))))
        
        self.coef, _ , self.rank, self.singular = lstsq(xTrain, yTrain)
        
        if self.fitIntercept:
            self.intercept = self.coef[-1]
            self.coef = self.coef[:-1]
        else:
            self.coef = self.coef
        
    def fitRidge(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        C = self.setting.getVals("C")
        
        _, nFeatures = xTrain.shape
        
        if self.fitIntercept:
            xOffset = np.mean(xTrain, axis=0)
            yOffset = np.mean(yTrain, axis=0)
            xTrain -= xOffset
            yTrain -= yOffset
            
        xTrain.flat[::nFeatures+1] += C
        A = np.dot(xTrain.T, xTrain)
        b = np.dot(xTrain.T, yTrain)
        
        self.coef = solve(A, b)
        
        if self.fitIntercept:
            self.intercept = yOffset-np.dot(xOffset.reshape(1,-1), self.coef)
            return self.coef, self.intercept
        else:
            return self.coef
    
    def fitLasso(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        from .lasso import celer, compute_norms_X_col, compute_Xw, dnorm_enet
        
        l1_ratio = 1.0
        
        C = self.setting.getVals("C")
        
        xTrain = np.asarray(xTrain, order='F')
        yTrain = np.asarray(yTrain, order='F')
        nSamples, nFeatures = xTrain.shape
        
        xDense = xTrain
        xData = np.empty([1], dtype=xTrain.dtype)
        xIndices = np.empty([1], dtype=np.int32)
        xIndptr = np.empty([1], dtype=np.int32)
        
        if self.fitIntercept:
            xOffset = np.mean(xTrain, axis=0)
            yOffset = np.mean(yTrain, axis=0)
            xTrain -= xOffset
            yTrain -= yOffset
            
            xSparseScaling = xOffset
        else:
            xSparseScaling = np.zeros(nFeatures, dtype=xTrain.dtype)
        
        norms_X_col = np.zeros(nFeatures, dtype=xDense.dtype)
        compute_norms_X_col(
            False, norms_X_col, nSamples, xDense, xData,
            xIndices, xIndptr, xSparseScaling)
        
        w = np.zeros(nFeatures, dtype=xDense.dtype)
        Xw = np.zeros(nSamples, dtype=xDense.dtype)
        compute_Xw(False, 0, Xw, w, yTrain.ravel(), xSparseScaling.any(), xDense,
                    xData, xIndices, xIndptr, xSparseScaling)
        theta = Xw.copy()
        
        weights = np.ones(nFeatures, dtype=xDense.dtype)
        positive = False
       
        skip = np.zeros(xTrain.shape[1], dtype=np.int32)
        dnorm = dnorm_enet(False, theta, w, xDense, xData, 
                           xIndices, xIndptr, skip, xSparseScaling, 
                           weights, xSparseScaling.any(), positive,
                           C, l1_ratio)
        
        theta /= max(dnorm / (C * l1_ratio), nSamples)
        
        #
        maxIters = self.setting.getVals("maxIter")
        maxEpochs = self.setting.getVals("maxEpoch")
        tl = self.setting.getVals("tol")
        p0 = self.setting.getVals("p0")

        #
        sol = celer(False, 0, xDense, xData, xIndices, 
                    xIndptr, xSparseScaling, yTrain.ravel(),
                    C, l1_ratio, w, Xw, 
                    theta, norms_X_col, weights,
                    max_iter=maxIters, max_epochs=maxEpochs,
                    p0=p0, verbose=0, use_accel=1, tol=tl, prune=True,
                    positive=positive)
        
        self.coef=sol[0]
        
        if self.fitIntercept:
            self.intercept=yOffset-np.dot(xOffset.reshape(1,-1), self.coef)
            return self.coef, self.intercept
        else:
            return self.coef