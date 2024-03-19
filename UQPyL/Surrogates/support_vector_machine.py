import numpy as np
from .libsvm import svm_fit, svm_predict, Parameter 
from typing import Tuple, Literal, Optional
from .surrogate_ABC import Scale_T, Surrogate
from ..Utility.polynomial_features import PolynomialFeatures
LINEAR = 0
POLYNOMIAL = 1
RBF = 2
SIGMOID = 3

def Fit(trainX, trainY, para, i, q):
    model=svm_fit(trainX, trainY, para, i)
    q.put(model)


class SVR(Surrogate):
    def __init__(self, 
                 scalers=(None, None), 
                 poly_feature: PolynomialFeatures=None,
                 C: float=1.0, epsilon: float=0.1, gamma: Optional[float]=0.0, coe0: float=0.0, degree: int=2,
                 kernel: Literal['linear', 'rbf', 'sigmoid', 'polynomial', 'precomputed']='rbf'):
        super().__init__(scalers, poly_feature)
        
        self.C=C
        self.epsilon=epsilon
        self.gamma=gamma
        self.coe0=coe0
        self.degree=degree
        self.kernel=kernel
        self.model=None
    ###############Interface Function#################
    def predict(self, predict_X: np.ndarray):
        
        predict_X=np.ascontiguousarray(predict_X).copy()
        predict_X=self.__X_transform__(predict_X)
        
        n_samples, _=predict_X.shape
        predict_Y=np.empty((n_samples,1))
        
        for i in range(n_samples):
            x=predict_X[i, :]
            predict_Y[i,0]=svm_predict(self.modelSetting, x)
            
        return self.__Y_inverse_transform__(predict_Y)
        
    def fit(self, trainX: np.ndarray,  trainY: np.ndarray):
        import threading
        from queue import Queue
        import time
        trainX=np.ascontiguousarray(trainX).copy()
        trainY=np.ascontiguousarray(trainY).copy()
        trainX, trainY=self.__check_and_scale__(trainX, trainY)
        
        _, n_feautes=trainX.shape
        
        nu=0.5
        ## Parameter: svm_type kernel_type degree gamma coef0 C nu p eps
        eps=0.0001
        par=Parameter(3, eval(self.kernel.upper()), self.degree, 100000,self.gamma, self.coe0, self.C, nu, self.epsilon, eps)     
        start=time.time()
        # for i in range(12):
        #     svm_fit(trainX, trainY.ravel(), par,i)
        # q=Queue()
        # threads=[]
        # for i in range(12):
        #     t=threading.Thread(target=Fit, args=(trainX, trainY.ravel(), par, i, q))
        #     t.start()
        #     threads.append(t)
        # for thread in threads:
        #     thread.join()
        # res = []
        # for _ in range(12):
        #     res.append(q.get())
        # end=time.time()
        # print(end-start)
        # self.modelSetting=svm_fit(trainX, trainY.ravel(), par)

        
    ###########################Attribute##############
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