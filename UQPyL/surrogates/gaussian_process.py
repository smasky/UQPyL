import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from typing import Callable, Tuple, Optional, Literal

from .surrogate_ABC import Surrogate, Scale_T
from .gp_kernels.Kernel import RBF, Matern, Kernel
from ..optimization import GA, Boxmin, MP_List, EA_List
from ..utility.model_selections import RandSelect
from ..utility.metrics import r2_score
from ..utility.scalers import Scaler
from ..utility.polynomial_features import PolynomialFeatures

class GPR(Surrogate):
    
    def __init__(self, kernel: Kernel, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 poly_feature: PolynomialFeatures=None,
                optimizer: Literal['Boxmin', 'GA']='Boxmin', n_restarts_optimizer: int=1,
                fitMode: Literal['likelihood', 'predictError']='likelihood',
                 alpha: float=1e-10):
        
        self.optimizer=optimizer
        self.fitMode=fitMode
        self.kernel=kernel
        self.alpha=alpha
        self.std=False
        self.n_restarts_optimizer=n_restarts_optimizer
        
        super().__init__(scalers=scalers, poly_feature=poly_feature)
        
###---------------------------------public function---------------------------------------###
    def fit(self, trainX: np.ndarray, trainY: np.ndarray):
        
        self.trainX, self.trainY=self.__check_and_scale__(trainX, trainY)
        
        if(isinstance(self.kernel, RBF) or isinstance(self.kernel, Matern)):
            
            if(self.fitMode=='likelihood'):
                self._fit_likelihood()
            elif(self.fitMode=='predictError'):
                self._fit_predictError()

        else:
            self._fit_pure_likelihood()
            
    def predict(self, predict_X: np.ndarray) -> np.ndarray:
        
        if self.X_scaler:
            predict_X=self.X_scaler.transform(predict_X)
        
        K_trans=self.kernel(predict_X,self.trainX)
        y_mean=K_trans @ self.alpha_
               
        V=solve_triangular(
            self.L_, K_trans.T, lower=True
        )
        
        if self.std:
            K=self.kernel(predict_X)
            y_var=np.diag(K).copy()
            y_var-=np.einsum("ij, ji->i", V.T, V)
            
            y_var[y_var<0]=0.0
            
            return y_mean, np.sqrt(y_var)
        return self.__Y_inverse_transform__(y_mean)
    
###--------------------------private functions--------------------###
    def _fit_predictError(self):
        
        TotalX=self.trainX
        TotalY=self.trainY
        
        #TODO using cross-validation KFold Method
        
        RS=RandSelect(20)
        train, test=RS.split(TotalX)
        
        testX=TotalX[test,:];testY=TotalY[test,:]
        trainX=TotalX[train,:];trainY=TotalY[train,:]
        
        if self.optimizer in MP_List:
            self.OPModel=eval(self.optimizer)(self.kernel.theta, self.kernel.theta_ub, self.kernel.theta_lb)
            def objFunc(theta):
                self.trainX=trainX
                self.trainY=trainY
                self._objfunc(theta, record=True)
                predictY=self.predict(self.X_scaler.inverse_transform(testX))
                return -1*r2_score(self.Y_scaler.inverse_transform(testY), predictY)
            bestTheta, bestObj=self.OPModel.run(objFunc)
            
        elif self.optimizer in EA_List:
            
            self.OPModel=eval(self.optimizer)(self.kernel.theta.size, np.log(self.kernel.theta_ub), np.log(self.kernel.theta_lb), 50)
            
            def objFunc(thetas):
                self.trainX=trainX
                self.trainY=trainY
                objs=np.zeros(thetas.shape[0])
                
                for i, theta in enumerate(thetas):
                    self._objfunc(np.power(np.e,theta),record=True)
                    predictY=self.predict(self.X_scaler.inverse_transform(testX))
                    objs[i]=-1*r2_score(self.Y_scaler.inverse_transform(testY), predictY)
                return objs.reshape(-1,1)
            
            self.OPFunc=objFunc
            bestObj=np.inf
            bestTheta=None
            for _ in range(self.n_restarts_optimizer):
                theta, obj=self.OPModel.run(self.OPFunc)
                if obj<bestObj:
                    bestTheta=theta
                    bestObj=obj
                    
            bestTheta, bestObj=self.OPModel.run(objFunc)
            bestTheta=np.power(np.e,bestTheta).ravel()
        
        self.trainX=TotalX
        self.trainY=TotalY
        self.theta=bestTheta
        self._objfunc(bestTheta, record=True)
                
        return self
    
    def _fit_pure_likelihood(self):
        
        self._objfunc(self.kernel.theta, record=True)  
        
    def _fit_likelihood(self):
            
        if self.optimizer in MP_List:
            self.OPModel=eval(self.optimizer)(self.kernel.theta, self.kernel.theta_ub, self.kernel.theta_lb)
            bestTheta, bestObj=self.OPModel.run(lambda theta:-self._objfunc(theta))
        elif self.optimizer in EA_List:
            
            self.OPModel=eval(self.optimizer)(self.kernel.theta.size, np.log(self.kernel.theta_ub), np.log(self.kernel.theta_lb), 50)
            def objFunc(thetas):
                objs=np.zeros(thetas.shape[0])
                for i,theta in enumerate(thetas):
                    objs[i]=self._objfunc(np.power(np.e,theta), record=True)
                return objs.reshape((-1,1))
            
            bestTheta, bestObj=self.OPModel.run(objFunc)
            bestTheta=np.power(np.e,bestTheta).ravel()
            
        #Prepare for prediction
        self._objfunc(bestTheta, record=True)        
        return self
    
    def _objfunc(self, theta: np.ndarray, record=False):
        """
            log_marginal_likelihood
        """
        self.kernel.theta=theta
        
        K=self.kernel(self.trainX)
        K[np.diag_indices_from(K)]+=self.alpha
        
        try:
            L=cholesky(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf
        
        alpha=cho_solve((L, True), self.trainY, check_finite=False)
        log_likelihood_dims= -0.5* np.einsum("ik,ik->k", self.trainY, alpha)
        log_likelihood_dims-=np.log(np.diag(L)).sum()
        log_likelihood_dims-=K.shape[0]/2 * np.log(2*np.pi)
        log_likelihood=np.sum(log_likelihood_dims)
        
        if record:
            self.L_=L
            self.alpha_=alpha
            self.theta=theta
            
        return log_likelihood
    
    
        
        
        
             
        
        
        
        
        
          




