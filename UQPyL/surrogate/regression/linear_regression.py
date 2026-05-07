import numpy as np
from scipy.linalg import lstsq, solve
from typing import Tuple, Literal, Optional, Union

from ..base import SurrogateABC
from ..scaler import Scaler
from ..poly import PolyFeature

class LinearRegression(SurrogateABC):
    '''
    Linear regression surrogate model.

    Supported loss types:
    - `Origin`: ordinary least squares
    - `Ridge`: L2-regularized regression
    - `Lasso`: L1-regularized regression

    Examples:
        >>> model = LinearRegression(lossType='Ridge')
        >>> model.fit(xTrain, yTrain)
        >>> yPred = model.predict(xPred)

    References:
        [1] A. E. Hoerl and R. W. Kennard, Ridge regression: Biased estimation for
            nonorthogonal problems, Technometrics, vol. 12, no. 1, pp. 55-67, 1970.
        [2] R. Tibshirani, Regression shrinkage and selection via the lasso,
            Journal of the Royal Statistical Society: Series B, vol. 58, no. 1,
            pp. 267-288, 1996.
    '''
    
    name = "LR"
    defaultTuneParameters = ()
    advancedTuneParameters = ("lossType", "C")
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                    polyFeature: PolyFeature = None, 
                        lossType: Literal['Origin', 'Ridge', 'Lasso'] = 'Origin',
                            fitIntercept: bool = True,
                                C: float=0.1, 
                                C_attr: Union[dict, None] = {'ub': 100, 'lb': 1e-5, 'type': 'float', 'log': True},
                                maxIter: int = 100, maxEpoch: int = 5e5, tolerance: float = 1e-3, p0: int = 10
                                ):

        super().__init__(scalers, polyFeature)
        
        self.fitIntercept = fitIntercept

        self.registerChoiceParameter("lossType", ["Origin", "Ridge", "Lasso"], owner="model")
        self.registerParameterApplier("lossType", self.setLossType)
        self.setting.set("C", C, C_attr)
        self.setting.set("maxIter", maxIter)
        self.setting.set("maxEpoch", maxEpoch)
        self.setting.set("tol", tolerance)
        self.setting.set("p0", p0)
        self.setLossType(lossType)
                
###---------------------------------public function---------------------------------------###
    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)

        if self.lossType == 'Origin':
            self.fitOrigin(xTrain, yTrain)
        elif self.lossType == 'Ridge':
            self.fitRidge(xTrain, yTrain)
        elif self.lossType == 'Lasso':
            self.fitLasso(xTrain, yTrain)
        else:
            raise ValueError('Using wrong model type!')
        
        return self

    def isParameterActive(self, name: str):
        if name == "C":
            return self.lossType in {"Ridge", "Lasso"}
        if name in {"maxIter", "maxEpoch", "tol", "p0"}:
            return self.lossType == "Lasso"
        return super().isParameterActive(name)

    def getDefaultTuneParameters(self, advanced: bool = False):
        params = list(self.defaultTuneParameters)
        if advanced:
            params.extend(self.advancedTuneParameters)
        return [name for name in params if self.isParameterActive(name) or name == "lossType"]

    def setLossType(self, lossType: str):
        if lossType not in {"Origin", "Ridge", "Lasso"}:
            raise ValueError("lossType must be one of ['Origin', 'Ridge', 'Lasso'].")
        self.lossType = lossType
        if "lossType" in self.setting.parVal:
            choiceInfo = self.setting.parSet["lossType"]
            self.setting.parVal["lossType"][:] = self.setting._normalize_choice_array(lossType, choiceInfo)
        self.resetFitState()
        return self
        
    def predict(self, xPred: np.ndarray, returnStd: bool = False,
                returnVar: bool = False) -> np.ndarray:
        self._normalize_predict_flags(returnStd, returnVar)
        self.requireFitted("coef", "intercept")
        
        xPred = self.__X_transform__(xPred)
        
        yPred = xPred @ self.fitState["coef"] + self.fitState["intercept"]
        yPred = yPred.reshape(-1,1)
        
        return self.__Y_inverse_transform__(yPred)
    
###--------------------------private functions----------------------------###
    def fitOrigin(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        if self.fitIntercept:
            xTrain = np.hstack((xTrain, np.ones((xTrain.shape[0], 1))))
        
        self.coef, _ , self.rank, self.singular = lstsq(xTrain, yTrain)
        
        if self.fitIntercept:
            intercept = self.coef[-1]
            coef = self.coef[:-1]
        else:
            coef = self.coef
            intercept = 0.0

        self.coef = coef
        self.intercept = intercept
        self.fitState["coef"] = coef
        self.fitState["intercept"] = intercept
        self.fitState["rank"] = self.rank
        self.fitState["singular"] = self.singular
        
    def fitRidge(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        C = self.setting.get("C")
        
        _, nFeatures = xTrain.shape
        
        if self.fitIntercept:
            xOffset = np.mean(xTrain, axis=0)
            yOffset = np.mean(yTrain, axis=0)
            xCentered = xTrain - xOffset
            yCentered = yTrain - yOffset
        else:
            xCentered = xTrain
            yCentered = yTrain
            
        A = np.dot(xCentered.T, xCentered)
        A.flat[::nFeatures + 1] += C
        b = np.dot(xCentered.T, yCentered)
        
        self.coef = solve(A, b)
        
        if self.fitIntercept:
            self.intercept = yOffset-np.dot(xOffset.reshape(1,-1), self.coef)
        else:
            self.intercept = 0.0

        self.fitState["coef"] = self.coef
        self.fitState["intercept"] = self.intercept
        self.fitState["rank"] = None
        self.fitState["singular"] = None
    
    def fitLasso(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        from .lasso import celer, compute_norms_X_col, compute_Xw, dnorm_enet
        
        l1_ratio = 1.0
        
        C = self.setting.get("C")
        
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
        maxIters = self.setting.get("maxIter")
        maxEpochs = self.setting.get("maxEpoch")
        tl = self.setting.get("tol")
        p0 = self.setting.get("p0")

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
        else:
            self.intercept = 0.0

        self.fitState["coef"] = self.coef
        self.fitState["intercept"] = self.intercept
        self.fitState["rank"] = None
        self.fitState["singular"] = None
