import numpy as np
from typing import Literal, Optional

from .core import svm_fit, svm_predict, Parameter 
from ..surrogateABC import Surrogate
from ...utility.polynomial_features import PolynomialFeatures
LINEAR = 0
POLYNOMIAL = 1
RBF = 2
SIGMOID = 3

class SVR(Surrogate):
    model=None
    def __init__(self, 
                 scalers=(None, None), 
                 polyFeature: PolynomialFeatures=None,
                 kernel: Literal['linear', 'rbf', 'sigmoid', 'polynomial']='rbf',
                 C: float=0.1, C_ub: float=1e3, C_lb: float=1e-5,
                 epsilon: float=0.1, epsilon_ub: float=1e3, epsilon_lb: float=1e-5,
                 gamma: float=1.0, gamma_ub: float=1e3, gamma_lb: float=1e-5,
                 coe0: float=0.0, coe0_ub: float=1e3, coe0_lb: float=1e-5,
                 degree: int=3, maxIter: int=1e5,  eps: float=0.001):
        
        super().__init__(scalers, polyFeature)
        
        self.setPara("C", C, C_lb, C_ub)
        self.setPara("epsilon", epsilon, epsilon_lb, epsilon_ub)
        self.setPara("gamma", gamma, gamma_lb, gamma_ub)
        
        if kernel=='sigmoid':
            self.setPara("coe0", coe0, coe0_lb, coe0_ub)
        
        if kernel=="polynomial":
            self.degree=degree
            
        self.kernel=kernel
        self.maxIter=maxIter
        self.eps=eps
        
###-----------------------public functions--------------------------###

    def predict(self, xPred: np.ndarray) -> np.ndarray:
        
        xPred=np.ascontiguousarray(xPred).copy()
        xPred=self.__X_transform__(xPred)
        
        nSample, _=xPred.shape
        predict_Y=np.empty((nSample,1))
        
        for i in range(nSample):
            x=xPred[i, :]
            predict_Y[i,0]=svm_predict(self.model, x)
            
        return self.__Y_inverse_transform__(predict_Y)
        
    def fit(self, xTrain: np.ndarray,  yTrain: np.ndarray):
        
        xTrain=np.ascontiguousarray(xTrain).copy()
        yTrain=np.ascontiguousarray(yTrain).copy()
        xTrain, yTrain=self.__check_and_scale__(xTrain, yTrain)
        
        C=self.getPara("C")
        gamma=self.getPara("gamma")
        epsilon=self.getPara("epsilon")
        coe0=self.getPara("coe0") if self.kernel=="sigmoid" else 0.0
        degree=self.degree if self.kernel=="polynomial" else 2
        ## Parameter: svm_type kernel_type degree maxIter gamma coef0 C nu p eps
        par=Parameter(int(3), int(eval(self.kernel.upper())), int(degree), int(self.maxIter), float(gamma), float(coe0), float(C), float(0.5), float(epsilon), float(self.eps))     
        self.model=svm_fit(xTrain, yTrain.ravel(), par)