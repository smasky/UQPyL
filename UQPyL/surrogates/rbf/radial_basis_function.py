import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lu, pinv
from typing import Tuple, Optional

from .kernels import BaseKernel, Cubic
from ..surrogateABC import Surrogate, Scale_T
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures

class RBF(Surrogate):
    '''
    Radial basis function network
    '''    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), polyFeature: PolynomialFeatures=None,
                 kernel: Optional[BaseKernel]=None, 
                 C_smooth: int=0, C_smooth_lb: int=1e-5, C_smooth_ub: int=1e5):
        
        super().__init__(scalers, polyFeature)
        
        self.setParameters("C_smooth", C_smooth, C_smooth_lb, C_smooth_ub)
        
        if (isinstance(kernel, BaseKernel)):
            kernel=kernel
        else:
            kernel=Cubic()
        
    def setKernel(self, kernel: BaseKernel):
        
        self.kernel=kernel
        
        
    def _get_tail_matrix(self, kernel: BaseKernel, train_X: np.ndarray):

        if(kernel.name=="Cubic" or kernel.name=="Thin_plate_spline"):
            tail_matrix=np.ones((self.n_samples,self.n_features+1))
            tail_matrix[:self.n_samples,:self.n_features]=train_X.copy()
            return tail_matrix
        elif (kernel.name=="Linear" or kernel.name=="Multiquadric"):
            tail_matrix=np.ones((self.n_samples,1))
            return tail_matrix
        else:
            return None
        
###--------------------------public functions----------------------------###

    def fit(self,train_X: np.ndarray,train_Y: np.ndarray):
        
        train_X, train_Y=self.__check_and_scale__(train_X,train_Y)
        
        A_Matrix=self.kernel.get_A_Matrix(train_X)+self.C_smooth
        P, L, U=lu(a=A_Matrix)
        L=np.dot(P,L)
        degree=self.kernel.get_degree(self.n_features)
        
        if(degree):
            bias=np.vstack((train_Y,np.zeros((degree,1))))
        else:
            bias=train_Y
        
        solve=np.dot(np.dot(pinv(U),pinv(L)),bias)

        if(degree):
            coe_h=solve[self.n_samples:,:]
        else:
            coe_h=0
        
        self.coe_h=coe_h
        self.coe_lambda=solve[:self.n_samples,:]
        self.train_X=train_X
        
    def predict(self, predict_X: np.ndarray):
        
        predict_X=self.__X_transform__(predict_X)
        
        dist=cdist(predict_X, self.train_X)
        temp1=np.dot(self.kernel.evaluate(dist),self.coe_lambda)
        temp2=np.zeros((temp1.shape[0],1))
        
        degree=self.kernel.get_degree(self.n_features)
        if(degree):
            if(degree>1):
                temp2=temp2+np.dot(predict_X,self.coe_h[:-1,:])
            if(degree>0):
                temp2=temp2+np.repeat(self.coe_h[-1:,:],temp1.shape[0],axis=0)
        
        return self.__Y_inverse_transform__(temp1+temp2)