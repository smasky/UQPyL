import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from typing import Tuple, Optional, Literal

from .kernel import BaseKernel, RBF
from ..util.boxmin import Boxmin
from ...problems import PracticalProblem
from ..surrogateABC import Surrogate
from ...optimization import Algorithm
from ...optimization.single_objective import GA
from ...utility.model_selections import RandSelect
from ...utility.metrics import r_square
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures

class GPR(Surrogate):
    
    name = "GPR"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 polyFeature: PolynomialFeatures = None,
                 kernel: BaseKernel = RBF(),
                 optimizer: Algorithm = 'Boxmin', n_restarts_optimizer: int = 0,
                 fitMode: Literal['likelihood', 'predictError', 'direct'] = 'likelihood',
                 C: float = 1e-9, C_ub: float = 1e-6, C_lb: float = 1e-12):
        
        super().__init__(scalers=scalers, polyFeature=polyFeature)
        
        self.setPara("C", C, C_lb, C_ub)
        
        self.fitMode = fitMode
        
        if isinstance(optimizer, Algorithm):
            # optimizer.verbose = False
            optimizer.saveFlag = False
            optimizer.logFlag = False
        else:
            optimizer = Boxmin()
            
        self.optimizer = optimizer
        
        self.kernel = kernel
        self.addSetting(kernel.setting)
        
        self.n_restarts_optimizer = n_restarts_optimizer
        
###---------------------------------public function---------------------------------------###
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain = self.__check_and_scale__(xTrain, yTrain)
        
        self.setKernel(self.kernel, xTrain.shape[1])
        
        if self.fitMode == 'likelihood':
            self._fitLikelihood(xTrain, yTrain)
            
        elif self.fitMode == 'predictError':
            self._fitPredictError(xTrain, yTrain)
        
        else:
            self._fitPure(xTrain, yTrain)
            
    def predict(self, xPred: np.ndarray, Output_std: bool=False):
        
        xPred = self.__X_transform__(xPred)
        
        K_trans = self.kernel(xPred, self.xTrain)
        y_mean = K_trans @ self.alpha_
               
        V = solve_triangular(
            self.L_, K_trans.T, lower=True
        )
        
        if Output_std:
            
            K = self.kernel(xPred)
            y_var = np.diag(K).copy()
            y_var -= np.einsum("ij, ji->i", V.T, V)
            y_var[y_var<0] = 0.0
            
            return y_mean, np.sqrt(y_var)
        
        return self.__Y_inverse_transform__(y_mean)
    
###--------------------------private functions--------------------###
    def _fitPredictError(self, xTrain, yTrain):
        
        tol_xTrain = np.copy(xTrain)
        tol_yTrain = np.copy(yTrain)
        
        #TODO cross-validation KFold Method
        
        RS = RandSelect(10)
        train, test = RS.split(tol_xTrain)
        
        xTest = tol_xTrain[test,:]; yTest = tol_yTrain[test,:]
        xTrain = tol_xTrain[train,:]; yTrain = tol_yTrain[train,:]
        
        self.xTrain = xTrain; self.yTrain = yTrain
        nameList = list(self.setting.parasValue.keys())
        
        paraInfos, ub, lb = self.setting.getParaInfos(nameList)
        nInput = ub.size #TODO
        
        if self.optimizer.type=="MP":
            
            def objFunc(varValue):
                
                self.assignPara(paraInfos, np.exp(varValue))
                obj = self._objfunc(xTrain, yTrain, record=True)
                if obj==-np.inf:
                    obj = obj*-1  
                else:
                    yPred = self.predict(self.__X_inverse_transform__(xTest))
                    obj = -1*r_square(self.__Y_inverse_transform__(yTest), yPred)
                
                return obj
            
            problem=PracticalProblem(objFunc, nInput=nInput, nOutput=1, ub=np.log(ub), lb=np.log(lb))
            
            bestDec, bestObj = self.optimizer.run(problem)
            
        elif self.optimizer.type=="EA":
            
            def objFunc(varValues):
                
                varValues = np.exp(varValues)
                objs = np.ones(varValues.shape[0])
                
                for i, varValue in enumerate(varValues):
                    
                    self.assignPara(paraInfos, varValue)

                    obj=self._objfunc(xTrain, yTrain, record=True)
                    if obj==-np.inf:
                        objs[i] = obj*-1
                        
                    else:
                        yPred = self.predict(self.__X_inverse_transform__(xTest))
                        objs[i] = -1*r_square(self.__Y_inverse_transform__(yTest), yPred)

                return objs.reshape( (-1, 1) )
            
            problem = PracticalProblem(objFunc, nInput, 1, np.log(ub), np.log(lb))
            res = self.optimizer.run(problem)
            bestDec, bestObj = res.bestDec, res.bestObj

            for _ in range(self.n_restarts_optimizer):
                res = self.optimizer.run(problem)
                dec, obj = res.bestDec, res.bestObj
                if obj < bestObj:
                    bestDec, bestObj = dec, obj
        #TODO     
        if bestObj>-0.99:
            self.xTrain = tol_xTrain; self.yTrain = tol_yTrain
        else:
            self.xTrain = xTrain; self.yTrain = yTrain
            
        self.assignPara(paraInfos, np.exp(bestDec))
        self._objfunc(self.xTrain, self.yTrain, record=True) #TODO
    
    def _fitPure(self, xTrain, yTrain):
        
        self.xTrain = xTrain; self.yTrain = yTrain
        
        self._objfunc( xTrain, yTrain, record=True )
        
    def _fitLikelihood(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        nameList = list(self.setting.parasValue.keys())
        
        paraInfos, ub, lb = self.setting.getParaInfos(nameList)
        nInput = ub.size
        
        if self.optimizer.type=="MP":
            
            def objFunc(varValue):

                self.assignPara(paraInfos, np.exp(varValue))
                
                return -self._objfunc(xTrain, yTrain, record=False)
                
            problem = PracticalProblem(objFunc, nInput=nInput, nOutput=1, ub=np.log(ub), lb=np.log(lb))
            bestDec, bestObj = self.optimizer.run(problem)
            
        elif self.optimizer.type=="EA":
            
            def objFunc(varValues):
                
                objs=np.zeros(varValues.shape[0])
                
                varValues=np.exp(varValues)
                
                for i, value in enumerate(varValues):
                    
                    self.assignPara(paraInfos, value)
                    
                    objs[i]=-1*self._objfunc(xTrain, yTrain, record=False)
                    
                return objs.reshape((-1, 1))
            
            problem=PracticalProblem(objFunc, nInput, 1, np.log(ub), np.log(lb))
            res=self.optimizer.run(problem)
            bestDec, bestObj=res.bestDec, res.bestObj
            
            for _ in range(self.n_restarts_optimizer):
                
                res=self.optimizer.run(problem)
                dec, obj=res.bestDec, res.bestObj
                
                if obj < bestObj:
                    bestDec, bestTheta=dec, obj
                    
        self.assignPara(paraInfos, np.exp(bestDec))
        
        #Prepare for prediction
        self.xTrain = xTrain; self.yTrain = yTrain
        self._objfunc(xTrain, yTrain, record=True)
        
    def _objfunc(self, xTrain, yTrain, record=False):
        """
            log_marginal_likelihood
        """
        
        K=self.kernel(xTrain)
        
        C=self.getPara("C")
        
        K[np.diag_indices_from(K)]+=C
        
        try:
            L=cholesky(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf
        
        alpha=cho_solve((L, True), yTrain, check_finite=False)
        log_likelihood_dims= -0.5* np.einsum("ik,ik->k", yTrain, alpha)
        log_likelihood_dims-=np.log(np.diag(L)).sum()
        log_likelihood_dims-=K.shape[0]/2 * np.log(2*np.pi)
        log_likelihood=np.sum(log_likelihood_dims)
        
        if record:
            self.L_=L
            self.alpha_=alpha

        return log_likelihood
    
    def setKernel(self, kernel: BaseKernel, N):
        
        self.kernel=kernel
        kernel.initialize(N)
        self.setting.mergeSetting(self.kernel.setting)
        
        
        
             
        
        
        
        
        
          




