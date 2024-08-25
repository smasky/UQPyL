import numpy as np
from .svr_ import svm_fit, svm_predict, Parameter 
from typing import Literal, Optional
from .surrogate_ABC import Surrogate
from ..utility.polynomial_features import PolynomialFeatures
LINEAR = 0
POLYNOMIAL = 1
RBF = 2
SIGMOID = 3

class SVR(Surrogate):
    def __init__(self, 
                 scalers=(None, None), 
                 poly_feature: PolynomialFeatures=None,
                 C: float=1.0, epsilon: float=0.1, 
                 gamma: Optional[float]=0.0, 
                 coe0: float=0.0, degree: int=2,
                 maxIter: int=100000, eps: float=0.001,
                 kernel: Literal['linear', 'rbf', 'sigmoid', 'polynomial']='rbf'):
        
        super().__init__(scalers, poly_feature)
        
        self.C=C
        self.epsilon=epsilon
        self.gamma=gamma
        self.coe0=coe0
        self.degree=degree
        self.kernel=kernel
        self.maxIter=maxIter
        self.eps=eps
        self.model=None
###-----------------------public functions--------------------------###

    def predict(self, predict_X: np.ndarray) -> np.ndarray:
        
        predict_X=np.ascontiguousarray(predict_X).copy()
        predict_X=self.__X_transform__(predict_X)
        
        n_samples, _=predict_X.shape
        predict_Y=np.empty((n_samples,1))
        
        for i in range(n_samples):
            x=predict_X[i, :]
            predict_Y[i,0]=svm_predict(self.model, x)
            
        return self.__Y_inverse_transform__(predict_Y)
        
    def fit(self, trainX: np.ndarray,  trainY: np.ndarray):
        
        trainX=np.ascontiguousarray(trainX).copy()
        trainY=np.ascontiguousarray(trainY).copy()
        trainX, trainY=self.__check_and_scale__(trainX, trainY)
        
        ## Parameter: svm_type kernel_type degree maxIter gamma coef0 C nu p eps
        par=Parameter(3, eval(self.kernel.upper()), self.degree, self.maxIter, self.gamma, self.coe0, self.C, 0.5, self.epsilon, self.eps)     
        self.model=svm_fit(trainX, trainY.ravel(), par)
        
###-----------------------attribute--------------------------###
    @property
    def C(self):
        return self.C_
    
    @C.setter
    def C(self, value):
        self.C_=value
    
    @property
    def epsilon(self):
        return self.epsilon_
    
    @epsilon.setter
    def epsilon(self, value):
        self.epsilon_=value
    
    @property
    def gamma(self):
        return self.gamma_
    
    @gamma.setter
    def gamma(self, value):
        self.gamma_=value
    
    @property
    def degree(self):
        return self.degree_
    
    @degree.setter
    def degree(self, value):
        self.degree_=value
    
    @property
    def coe0(self):
        return self.coe0_
    
    @coe0.setter
    def coe0(self, value):
        self.coe0_=value

    @property
    def model(self):
        return self.model_
    
    @model.setter
    def model(self, value):
        self.model_=value