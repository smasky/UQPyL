import numpy as np
from typing import Literal, Optional, Tuple, Union

from .core import svm_fit, svm_predict, Parameter 
from ..base import SurrogateABC
from ...util.scaler import Scaler
from ...util.poly import PolyFeature

class SVR(SurrogateABC):
    '''
    Support Vector Regression(SVR)
    -----------------------------
    This class is a interface of libsvm library from Lin Chih-Jen Professor in National Taiwan University.
    For regression problems, the epsilon-SVR or nu-SVR is used here.
    
    References:
        [1] C. C. Chang and C. J. Lin, "LIBSVM: A library for support vector machines", 2015.
    
    Methods:
        predict(xPred): 
            Predicts the output of the surrogate model for a given input.
            - xPred: np.ndarray
                The input to predict the output for.
        fit(xTrain, yTrain):
            Fits the surrogate model to the training data.
            - xTrain: np.ndarray
                The input training data.
            - yTrain: np.ndarray
                The output training data.
    '''
    
    name = "SVR"
    _KERNEL_CODES = {
        'linear': 0,
        'polynomial': 1,
        'rbf': 2,
        'sigmoid': 3,
    }
    _SYMBOL_CODES = {
        'epsilon-SVR': 3,
        'nu-SVR': 4,
    }
    defaultTuneParameters = ("C", "gamma", "epsilon")
    advancedTuneParameters = ("kernel", "symbol", "nu", "coe0", "degree")
    
    def __init__(self, 
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None), 
                 polyFeature: PolyFeature = None,
                 symbol: Literal['epsilon-SVR', 'nu-SVR'] = 'epsilon-SVR',
                 kernel: Literal['linear', 'rbf', 'sigmoid', 'polynomial'] = 'rbf',
                 nu: float = 0.5, nu_attr: Union[dict, None] = {'ub': 1e3, 'lb': 1e-5, 'type': 'float', 'log': True},
                 C: float = 0.1, C_attr: Union[dict, None] = {'ub': 1e3, 'lb': 1e-5, 'type': 'float', 'log': True},
                 epsilon: float = 0.1, epsilon_attr: Union[dict, None] = {'ub': 1e3, 'lb': 1e-5, 'type': 'float', 'log': True},
                 gamma: float = 1.0, gamma_attr: Union[dict, None] = {'ub': 1e3, 'lb': 1e-5, 'type': 'float', 'log': True},
                 coe0: float = 0.1, coe0_attr: Union[dict, None] = {'ub': 1e3, 'lb': 1e-5, 'type': 'float', 'log': True},
                 degree: int=3, maxIter: int=1e5,  eps: float=0.001):
        '''
        Initialize the SVR surrogate model.
        
        :param symbol: Literal['epsilon-SVR', 'nu-SVR']
            The type of SVR to use.
        :param kernel: Literal['linear', 'rbf', 'sigmoid', 'polynomial']
            The kernel to use. 
            'linear' -> u'*v
            'rbf' -> exp(-gamma*|u-v|^2)
            'sigmoid' -> tanh(gamma*u'*v + coef0)
            'polynomial' -> (gamma*u'*v + coef0)^degree
        :param C: float
            The regularization parameter of epsilon-SVR or nu-SVR.
        :param nu: float
            The nu parameter of nu-SVR.
        :param epsilon: float
            The epsilon parameter in loss function of epsilon-SVR.
        :param gamma: float
            The gamma parameter of rbf, sigmoid, polynomial kernel.
        :param coe0: float
            The coef0 parameter of sigmoid, polynomial kernel.
        :param degree: int
            The degree parameter of polynomial kernel.
        :param maxIter: int
            The maximum number of iterations.
        :param eps: float
            The tolerance of the stopping criterion.
        '''
        super().__init__(scalers, polyFeature)
        
        
        if symbol not in self._SYMBOL_CODES:
            raise ValueError(f"symbol must be in ['epsilon-SVR', 'nu-SVR'], but got {symbol}")
        self.symbolName = symbol
        self.symbol = self._SYMBOL_CODES[symbol]
        
        kernel = kernel.lower()
        if kernel in self._KERNEL_CODES:
            self.kernelName = kernel
            self.kernel = self._KERNEL_CODES[kernel]
        else:
            raise ValueError(f"kernel must be in ['linear', 'rbf', 'sigmoid', 'polynomial'], but got {kernel}")
        
        self.innerModel = None
        self.registerChoiceParameter("symbol", list(self._SYMBOL_CODES.keys()), owner="model")
        self.registerChoiceParameter("kernel", list(self._KERNEL_CODES.keys()), owner="model")
        self.registerParameterApplier("symbol", self.setSymbol)
        self.registerParameterApplier("kernel", self.setKernel)
        
        self.setting.setPara("C", C, C_attr)
        self.setting.setPara("epsilon", epsilon, epsilon_attr)
        self.setting.setPara("gamma", gamma, gamma_attr)
        self.setting.setPara("coe0", coe0, coe0_attr)
        self.setting.setPara("degree", degree)
        self.setting.setPara("maxIter", maxIter)
        self.setting.setPara("eps", eps)
        self.setting.setPara("nu", nu, nu_attr)
        self.setSymbol(symbol)
        self.setKernel(kernel)
        
###-----------------------public functions--------------------------###

    def predict(self, xPred: np.ndarray, returnStd: bool = False,
                returnVar: bool = False):
        self._normalize_predict_flags(returnStd, returnVar)
        self.requireFitted("innerModel")
        
        xPred = np.ascontiguousarray(xPred).copy()
        xPred = self.__X_transform__(xPred)
        
        nSample, _ = xPred.shape
        predict_Y = np.empty((nSample,1))
        
        for i in range(nSample):
            x = xPred[i, :]
            predict_Y[i, 0] = svm_predict(self.innerModel, x)
            
        return self.__Y_inverse_transform__(predict_Y)

    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)

        xTrain = np.ascontiguousarray(xTrain).copy()
        yTrain = np.ascontiguousarray(yTrain).copy()
        par = self._build_parameter()
        self.innerModel = svm_fit(xTrain, yTrain.ravel(), par)
        self.fitState["innerModel"] = self.innerModel
        self.fitState["symbol"] = self.symbolName
        self.fitState["kernel"] = self.kernelName
        return self

    def fitHyper(self, xTrain: np.ndarray, yTrain: np.ndarray):
        return self.fitModel(xTrain, yTrain)

    def isParameterActive(self, name: str):
        if name == "epsilon":
            return self.symbolName == "epsilon-SVR"
        if name == "nu":
            return self.symbolName == "nu-SVR"
        if name == "gamma":
            return self.kernelName in {"rbf", "sigmoid", "polynomial"}
        if name == "coe0":
            return self.kernelName in {"sigmoid", "polynomial"}
        if name == "degree":
            return self.kernelName == "polynomial"
        if name in {"maxIter", "eps"}:
            return False
        return super().isParameterActive(name)

    def getDefaultTuneParameters(self, advanced: bool = False):
        params = list(self.defaultTuneParameters)
        if advanced:
            params.extend(self.advancedTuneParameters)
        return [name for name in params if self.isParameterActive(name) or name in {"kernel", "symbol"}]

    def setSymbol(self, symbol: str):
        if symbol not in self._SYMBOL_CODES:
            raise ValueError(f"symbol must be in ['epsilon-SVR', 'nu-SVR'], but got {symbol}")
        self.symbolName = symbol
        self.symbol = self._SYMBOL_CODES[symbol]
        self.resetFitState()
        return self

    def setKernel(self, kernel: str):
        kernel = kernel.lower()
        if kernel not in self._KERNEL_CODES:
            raise ValueError(f"kernel must be in ['linear', 'rbf', 'sigmoid', 'polynomial'], but got {kernel}")
        self.kernelName = kernel
        self.kernel = self._KERNEL_CODES[kernel]
        self.resetFitState()
        return self

    def _build_parameter(self):
        nu = self.setting.getVals("nu")
        C = self.setting.getVals("C")
        gamma = self.setting.getVals("gamma") if self.isParameterActive("gamma") else 0.0
        epsilon = self.setting.getVals("epsilon") if self.isParameterActive("epsilon") else 0.0
        coe0 = self.setting.getVals("coe0") if self.isParameterActive("coe0") else 0.0
        degree = self.setting.getVals("degree") if self.isParameterActive("degree") else 2
        maxIter = self.setting.getVals("maxIter")
        eps = self.setting.getVals("eps")
        return Parameter(
            int(self.symbol), int(self.kernel), int(degree), int(maxIter),
            float(gamma), float(coe0), float(C), float(nu), float(epsilon), float(eps)
        )
