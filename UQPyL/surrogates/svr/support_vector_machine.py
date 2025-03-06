import numpy as np
from typing import Literal, Optional, Tuple, Union

from .core import svm_fit, svm_predict, Parameter 
from ..surrogateABC import Surrogate
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures

class SVR(Surrogate):
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
    
    def __init__(self, 
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None), 
                 polyFeature: PolynomialFeatures = None,
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
        
        
        if symbol in ['epsilon-SVR', 'nu-SVR']:
            self.symbol = 3 if symbol == 'epsilon-SVR' else 4
        else:
            raise ValueError(f"symbol must be in ['epsilon-SVR', 'nu-SVR'], but got {symbol}")
        
        if kernel.lower() in ['linear','polynomial', 'rbf', 'sigmoid']:    
            self.kernel = 0 if kernel.lower() == 'linear' else 1 if kernel.lower() == 'polynomial' else 2 if kernel.lower() == 'rbf' else 3
        else:
            raise ValueError(f"kernel must be in ['linear', 'rbf', 'sigmoid', 'polynomial'], but got {kernel}")
        
        self.innerModel = None
        
        self.setting.setPara("C", C, C_attr)
        self.setting.setPara("epsilon", epsilon, epsilon_attr)
        self.setting.setPara("gamma", gamma, gamma_attr)
        self.setting.setPara("coe0", coe0, coe0_attr)
        self.setting.setPara("degree", degree)
        self.setting.setPara("maxIter", maxIter)
        self.setting.setPara("eps", eps)
        self.setting.setPara("nu", nu, nu_attr)
        
###-----------------------public functions--------------------------###

    def predict(self, xPred: np.ndarray):
        
        xPred = np.ascontiguousarray(xPred).copy()
        xPred = self.__X_transform__(xPred)
        
        nSample, _ = xPred.shape
        predict_Y = np.empty((nSample,1))
        
        for i in range(nSample):
            x = xPred[i, :]
            predict_Y[i, 0] = svm_predict(self.innerModel, x)
            
        return self.__Y_inverse_transform__(predict_Y)
        
    def fit(self, xTrain: np.ndarray,  yTrain: np.ndarray):
        
        xTrain = np.ascontiguousarray(xTrain).copy()
        yTrain = np.ascontiguousarray(yTrain).copy()
        xTrain, yTrain = self.__check_and_scale__(xTrain, yTrain)
        
        nu = self.setting.getVals("nu")
        C = self.setting.getVals("C")
        gamma = self.setting.getVals("gamma")
        epsilon = self.setting.getVals("epsilon")
        coe0 = self.setting.getVals("coe0") if self.kernel in ['sigmoid', 'polynomial'] else 0.0
        degree = self.setting.getVals("degree") if self.kernel in ['polynomial'] else 2
        maxIter = self.setting.getVals("maxIter")
        eps = self.setting.getVals("eps")
        ## Parameter: svm_type kernel_type degree maxIter gamma coef0 C nu p eps
        par = Parameter(int(self.symbol), int(self.kernel), int(degree), int(maxIter), float(gamma), float(coe0), float(C), float(nu), float(epsilon), float(eps))     
        self.innerModel = svm_fit(xTrain, yTrain.ravel(), par)
    
    def _fitPure(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain = np.ascontiguousarray(xTrain).copy()
        yTrain = np.ascontiguousarray(yTrain).copy()
        
        nu = self.setting.getVals("nu")
        C = self.setting.getVals("C")
        gamma = self.setting.getVals("gamma")
        epsilon = self.setting.getVals("epsilon")
        coe0 = self.setting.getVals("coe0") if self.kernel in ['sigmoid', 'polynomial'] else 0.0
        degree = self.setting.getVals("degree") if self.kernel in ['polynomial'] else 2
        maxIter = self.setting.getVals("maxIter")
        eps = self.setting.getVals("eps")
        ## Parameter: svm_type kernel_type degree maxIter gamma coef0 C nu p eps
        par = Parameter(int(self.symbol), int(self.kernel), int(degree), int(maxIter), float(gamma), float(coe0), float(C), float(nu), float(epsilon), float(eps))     
        self.innerModel = svm_fit(xTrain, yTrain.ravel(), par)