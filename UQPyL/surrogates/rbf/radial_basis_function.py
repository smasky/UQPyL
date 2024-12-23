import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lu, pinv
from typing import Tuple, Optional, Literal

from .kernel import BaseKernel, Cubic
from ..surrogateABC import Surrogate
from ...utility.metrics import r_square
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures
from ...utility.model_selections import RandSelect

class RBF(Surrogate):
    '''
    Radial basis function network
    '''   
    
    name = "RBF"
     
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), polyFeature: PolynomialFeatures=None,
                 kernel: Optional[BaseKernel]=Cubic(), 
                 C_smooth: int=0.0, C_smooth_lb: int=1e-5, C_smooth_ub: int=1e5):
        
        super().__init__(scalers, polyFeature)
        
        self.setPara("C_smooth", C_smooth, C_smooth_lb, C_smooth_ub)
        
        self.kernel = kernel
        self.setting.mergeSetting(kernel.setting)
        
        
    def setKernel(self, kernel: BaseKernel):
        
        self.kernel = kernel
        self.setting.mergeSetting(self.kernel.setting)
    
    def _get_tail_matrix(self, kernel: BaseKernel, train_X: np.ndarray):

        if(kernel.name == "Cubic" or kernel.name == "Thin_plate_spline"):
            tail_matrix = np.ones((self.n_samples,self.n_features+1))
            tail_matrix[:self.n_samples,:self.n_features]=train_X.copy()
            return tail_matrix
        
        elif (kernel.name == "Linear" or kernel.name == "Multiquadric"):
            tail_matrix = np.ones((self.n_samples,1))
            return tail_matrix
        else:
            return None
    
    def _fitPure(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        nSample, nFeature= xTrain.shape
        
        C_smooth=self.getPara("C_smooth")
        
        A_Matrix=self.kernel.get_A_Matrix(xTrain)+C_smooth
        
        P, L, U=lu(a=A_Matrix)
        L=np.dot(P,L)
        degree=self.kernel.get_degree(nFeature)
        
        if(degree):
            bias=np.vstack((yTrain, np.zeros((degree,1))))
        else:
            bias=yTrain
        
        solve=np.dot(np.dot(pinv(U), pinv(L)), bias)

        if(degree):
            coe_h=solve[nSample:, :]
        else:
            coe_h=0
        
        self.coe_h=coe_h
        self.coe_lambda=solve[:nSample, :]
        self.xTrain=xTrain
    
    # def _fitPredictError(self, xTrain: np.ndarray, yTrain: np.ndarray):
    #     tol_xTrain = np.copy(xTrain)
    #     tol_yTrain = np.copy(yTrain)
        
    #     RS = RandSelect(10)
    #     train, test = RS.split(tol_xTrain)
        
    #     xTest = tol_xTrain[test,:]; yTest = tol_yTrain[test,:]
    #     xTrain = tol_xTrain[train,:]; yTrain = tol_yTrain[train,:]
        
    #     self.xTrain = xTrain; self.yTrain = yTrain
        
    #     nameList = list(self.setting.parasValue.keys())
        
    #     paraInfos, ub, lb = self.setting.getParaInfos(nameList)
    #     nInput = ub.size #TODO
        
    #     def objFunc(varValues):
            
    #         varValues = np.exp(varValues)
    #         objs = np.ones(varValues.shape[0])
            
    #         for i, varValue in enumerate(varValues):
                    
    #                 self.assignPara(paraInfos, varValue)

    #                 obj=self._fitPure(xTrain, yTrain)
    #                 if obj==-np.inf:
    #                     objs[i] = obj*-1
                        
    #                 else:
    #                     yPred = self.predict(self.__X_inverse_transform__(xTest))
    #                     objs[i] = -1*r_square(self.__Y_inverse_transform__(yTest), yPred)

    #         return objs.reshape( (-1, 1) )
        
###--------------------------public functions----------------------------###
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain=self.__check_and_scale__(xTrain, yTrain)
        
        self._fitPure(xTrain, yTrain)
          
    def predict(self, xPred: np.ndarray):
        
        _, nFeature= xPred.shape
        
        xPred=self.__X_transform__(xPred)
        
        dist=cdist(xPred, self.xTrain)
        temp1=np.dot(self.kernel.evaluate(dist), self.coe_lambda)
        temp2=np.zeros((temp1.shape[0],1))
        
        degree=self.kernel.get_degree(nFeature)
        if(degree):
            if(degree>1):
                temp2=temp2+np.dot(xPred, self.coe_h[:-1,:])
            if(degree>0):
                temp2=temp2+np.repeat(self.coe_h[-1:,:], temp1.shape[0],axis=0)
        
        return self.__Y_inverse_transform__(temp1+temp2)
    
    def getParaList(self):
        
        return list(self.setting.parasValue.keys())