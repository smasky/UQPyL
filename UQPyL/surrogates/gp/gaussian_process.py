import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from typing import Tuple, Optional, Literal

from .kernel import BaseKernel, RBF
from ..util.boxmin import Boxmin
from ...problems import Problem
from ..surrogateABC import Surrogate
from ...optimization import Algorithm
from ...optimization.single_objective import GA
from ...utility.data_selections import RandSelect
from ...utility.metrics import r_square
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures

class GPR(Surrogate):
    
    name = "GPR"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                    polyFeature: PolynomialFeatures = None,
                        kernel: BaseKernel = RBF(),
                            optimizer: Algorithm = 'Boxmin', nRestartTimes: int = 0,
                                    C: float = 1e-9,
                                    C_attr: dict = {'ub': 1e-6, 'lb':1e-12, 
                                                        'type': 'float', 
                                                        'log': 'True'}):
        
        super().__init__(scalers=scalers, polyFeature=polyFeature)
        
        self.kernel = None
        
        self.setting.setPara("C", C, C_attr)
        
        if isinstance(optimizer, Algorithm):
            optimizer.verboseFlag = False
            optimizer.saveFlag = False
            optimizer.logFlag = False
        else:
            optimizer = Boxmin()
            
        self.optimizer = optimizer
        
        self.setKernel(kernel)
        
        self.nRes = nRestartTimes
        
###---------------------------------public function---------------------------------------###
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain = self.__check_and_scale__(xTrain, yTrain)
        
        # 定义阈值
        # threshold = 1

        # # 去除重复的行
        # unique_index = []
        # for i, row in enumerate(xTrain):
        #     if not any(np.allclose(row, xTrain[j], atol=threshold) for j in unique_index):
        #         unique_index.append(i)
                
        # xTrain = xTrain[unique_index]
        # yTrain = yTrain[unique_index]
        
        self.kernel.initialize( xTrain.shape[1] )
        
        self._fitLikelihood( xTrain, yTrain )
            
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
    def _fitPure(self, xTrain, yTrain):
        
        self.xTrain = xTrain; self.yTrain = yTrain
        
        self._objfunc( xTrain, yTrain, record=True )
        
    def _fitLikelihood(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        nameList = self.getParaList()
        
        paraInfos, ub, lb = self.setting.getParaInfos(nameList)
        
        nInput = ub.size
        
        if self.optimizer.type == "MP":
            
            def objFunc(varValue):

                self.setting.setVals(paraInfos, varValue)
                
                return self._objfunc(xTrain, yTrain, record = False)
                
            problem = Problem(nInput = nInput, nOutput = 1, ub = ub, lb = lb, 
                                objFunc = objFunc)
            
            bestDecs, bestObj = self.optimizer.run(problem)
        
        elif self.optimizer.type == "EA":
            
            def objFunc(varValues):
                
                objs = np.zeros(varValues.shape[0])
                
                # varValues[0, :] = np.array([-26.22] + [11.513]*15)
                
                for i, value in enumerate(varValues):
                    
                    self.setting.setVals(paraInfos, value)
                    
                    objs[i] = self._objfunc(xTrain, yTrain, record=False)
                    
                return objs.reshape( (-1, 1) )
            
            problem = Problem(nInput, 1, ub, lb, objFunc = objFunc)
            
            res = self.optimizer.run(problem)
            bestDecs, bestObj = res.bestDecs, res.bestObjs
            
            for _ in range(self.nRes):
                
                res = self.optimizer.run(problem)
                dec, obj = res.bestDecs, res.bestObjs
                
                if obj < bestObj:
                    bestDec, bestTheta = dec, obj
                    
        self.setting.setVals(paraInfos, bestDecs.ravel() )
        
        #Prepare for prediction
        self.xTrain = xTrain; self.yTrain = yTrain
        obj = self._objfunc(xTrain, yTrain, record=True)
        print(obj)
        
        
    def _objfunc(self, xTrain, yTrain, record=False):
        """
            log_marginal_likelihood
        """
        
        K = self.kernel(xTrain)
        
        C = self.setting.getVals("C")
        
        K[np.diag_indices_from(K)] += C
        
        try:
            L = cholesky(K, lower = True, check_finite = False)
        except np.linalg.LinAlgError as e:
            K[np.diag_indices_from(K)] += 1e-6
            L = cholesky(K, lower = True, check_finite = False)
        
        alpha = cho_solve((L, True), yTrain, check_finite=False)
        log_likelihood_dims =  -0.5* np.einsum("ik,ik->k", yTrain, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0]/2 * np.log(2*np.pi)
        log_likelihood = np.sum(log_likelihood_dims)
        
        if record:
            self.L_ = L
            self.alpha_ = alpha

        return log_likelihood
    
    def setKernel(self, kernel: BaseKernel):
        
        if self.kernel is not None:
            self.setting.removeSetting(self.kernel.setting) 
        
        self.kernel = kernel
        self.setting.mergeSetting(self.kernel.setting)
        self.kernel.setting = self.setting