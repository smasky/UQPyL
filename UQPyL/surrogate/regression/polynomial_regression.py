import numpy as np
from typing import Literal, Tuple, Optional, Union

from .linear_regression import LinearRegression
from ..scaler import Scaler
from ..poly import PolyFeature

class PolynomialRegression(LinearRegression):
    
    """
    Polynomial regression surrogate model.

    This model augments the input space with polynomial features, then fits
    an underlying linear, ridge, or lasso regression model.

    Examples:
        >>> model = PolynomialRegression(degree=2, lossType='Origin')
        >>> model.fit(xTrain, yTrain)
        >>> yPred = model.predict(xPred)
    """
    
    name = "PR"
    defaultTuneParameters = ("degree",)
    advancedTuneParameters = ("lossType", "C", "onlyInteraction")
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                degree: int = 2, degree_attr: Union[dict, None] = {'ub': 3, 'lb': 1, 'type': 'int', 'log': False},
                onlyInteraction: bool = False,
                lossType: Literal['Origin', 'Ridge', 'Lasso'] = 'Origin',
                fitIntercept: bool = True,
                C: float=0.1, C_attr: Union[dict, None] = {'ub': 100, 'lb': 1e-5, 'type': 'float', 'log': True},
                maxIter: int = 100, maxEpoch: int = 5e5, tolerance: float = 1e-3, p0: int = 10):
        
        super().__init__(scalers = scalers, polyFeature = None,
                         lossType = lossType, fitIntercept = fitIntercept, 
                         C = C, C_attr = C_attr, maxIter = maxIter, 
                         maxEpoch = maxEpoch, tolerance = tolerance, p0 = p0)
        
        self.degree = degree
        self.fitIntercept = fitIntercept
        self.onlyInteraction = onlyInteraction
        self.polyFeatureBuilder = PolyFeature(
            degree=degree,
            includeBias=False,
            onlyInteraction=onlyInteraction,
        )

        self.registerParameterApplier("degree", self.setDegree)
        self.registerChoiceParameter("lossType", ["Origin", "Ridge", "Lasso"], owner="model")
        self.registerChoiceParameter("onlyInteraction", [False, True], owner="model")
        self.registerParameterApplier("lossType", self.setLossType)
        self.registerParameterApplier("onlyInteraction", self.setOnlyInteraction)
        self.setting.set("degree", degree, degree_attr)
        self.setting.set("C", C, C_attr)
        self.setting.set("maxIter", maxIter)
        self.setting.set("maxEpoch", maxEpoch)
        self.setting.set("tol", tolerance)
        self.setting.set("p0", p0)
        self.setLossType(lossType)
        self.setOnlyInteraction(onlyInteraction)
        
###------------------------public functions-----------------------------###
    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)

        xTrain = self.polynomialFeatures(xTrain)
        
        if self.lossType == 'Origin':
            self.fitOrigin(xTrain, yTrain)
        elif self.lossType == 'Ridge':
            self.fitRidge(xTrain, yTrain)
        elif self.lossType == 'Lasso':
            self.fitLasso(xTrain, yTrain)
        else:
            raise ValueError('Using wrong model type!')

        return self
        
    def predict(self, xPred: np.ndarray, returnStd: bool = False,
                returnVar: bool = False) -> np.ndarray:
        self._normalize_predict_flags(returnStd, returnVar)
        self.requireFitted("coef", "intercept")
        
        xPred = self.__X_transform__(xPred)
        xPred = self.polynomialFeatures(xPred)
        
        yPred = xPred @ self.fitState["coef"] + self.fitState["intercept"]
            
        yPred = yPred.reshape(-1,1)
        
        return self.__Y_inverse_transform__(yPred)

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
        return [name for name in params if self.isParameterActive(name) or name in {"lossType", "onlyInteraction"}]

    def polynomialFeatures(self, xTrain: np.ndarray):
        return self.polyFeatureBuilder.transform(xTrain)

    def setDegree(self, degree: int):
        self.degree = int(degree)
        self.polyFeatureBuilder.degree = self.degree
        if "degree" in self.setting.parVal:
            self.setting.parVal["degree"][:] = self.degree
        self.resetFitState()
        return self

    def setLossType(self, lossType: str):
        if lossType not in {"Origin", "Ridge", "Lasso"}:
            raise ValueError("lossType must be one of ['Origin', 'Ridge', 'Lasso'].")
        self.lossType = lossType
        if "lossType" in self.setting.parVal:
            choiceInfo = self.setting.parSet["lossType"]
            self.setting.parVal["lossType"][:] = self.setting._normalize_choice_array(lossType, choiceInfo)
        self.resetFitState()
        return self

    def setOnlyInteraction(self, onlyInteraction: bool):
        self.onlyInteraction = bool(onlyInteraction)
        self.polyFeatureBuilder.onlyInteraction = self.onlyInteraction
        if "onlyInteraction" in self.setting.parVal:
            choiceInfo = self.setting.parSet["onlyInteraction"]
            self.setting.parVal["onlyInteraction"][:] = self.setting._normalize_choice_array(self.onlyInteraction, choiceInfo)
        self.resetFitState()
        return self
