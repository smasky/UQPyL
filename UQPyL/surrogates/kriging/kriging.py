import numpy as np
from scipy.linalg import LinAlgError, cholesky, qr, lstsq
from scipy.spatial.distance import pdist
from typing import Literal, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..surrogateABC import Surrogate
from .kernel import BaseKernel, Guass
from ...utility.metrics import r_square
from ...optimization.mathematics import Boxmin
from ...utility.model_selections import RandSelect
from ...utility.scalers import Scaler, StandardScaler
from ...utility.polynomial_features import PolynomialFeatures
from ...problems import PracticalProblem

####---------------------regression functions--------------------###
def regrpoly0(S):
    
    n_sample, _ = S.shape
    return np.ones((n_sample, 1))

def regrpoly1(S):
    
    n_sample, _ = S.shape
    return np.hstack((np.ones((n_sample, 1)), S))

def regrpoly2(S):
    
    n_sample, n_feature = S.shape
    nn = int((n_feature + 1) * (n_feature + 2) / 2)
    F = np.hstack((np.ones((n_sample, 1)), S, np.zeros((n_sample, nn - n_feature - 1))))
    j = n_feature + 1
    q = n_feature

    for k in np.arange(1, n_feature + 1):
        F[:, j + np.arange(q)] = np.tile(S[:, (k - 1):k],
                                            (1, q)) * S[:, np.arange(k - 1, n_feature)]
        j += q;q -= 1
    return F

class KRG(Surrogate):
    """
    A Kriging implementation based on python env includes the new training method(prediction error), 
    from the DACE toolbox(MATLAB).
    
    parameters:
    
    theta0: initial theta
    lb: the low bound of the theta
    ub: the up bound of the theta
    
    regression: type of regression functions, containing:
                *'poly0'
                *'poly1'
                *'poly2'
    
    correlation: the correlation function, only 'corrgauss'
    
    optimizer: the method used to find the optimal theta for current data, containing:
                * all Single Objective Algorithm
                *'Boxmin'
    
    n_restart_optimize: the times of using evolutionary algorithms to optimize theta 
    
    fitMode: the objective function used to evaluate the performance of the theta, containing:
                *'likelihood' origin way
                *'predictError' new way
    
    normalized: the sign to normalize input data(x, y) or not
    
    Scale_type: the normalized method, containing:
            *'StandardScaler'
            *'MaxminScaler'
            
    """
    def __init__(self, 
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 polyFeature: PolynomialFeatures=None,
                 kernel: BaseKernel= Guass(),
                 regression: Literal['poly0','poly1','poly2']='poly0',
                 optimizer: 'Algorithm' = Boxmin(), 
                 fitMode: Literal['likelihood', 'predictError']='likelihood',
                 n_restart_optimize: int=1):
        
        super().__init__(scalers, polyFeature)
    
        self.optimizer=optimizer
        self.fitMode=fitMode
        self.n_restart_optimize=n_restart_optimize
        
        if not isinstance(kernel, BaseKernel):
            raise ValueError("The kernel must be the instance of Krg_Kernel")
        
        self.setKernel(kernel)
        
        if(regression=='poly0'):
            self.regrFunc=regrpoly0
        elif(regression=='poly1'):
            self.regrFunc=regrpoly1
        elif(regression=='poly2'):
            self.regrFunc=regrpoly2
        
###-------------------------------public function-----------------------------###
    def predict(self, xPred: np.ndarray, only_value=True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        xPred=self.__X_transform__(xPred)
        
        nSample, _= self.xTrain.shape
        
        nPred, nFeature=xPred.shape
        
        dx = np.zeros( (nPred * nSample, nFeature) )
        
        kk = np.arange( nSample )

        for k in np.arange(nPred):
            dx[kk, :] = xPred[k, :] - self.xTrain
            kk = kk + nSample
        
        F=self.regrFunc(xPred)
        
        r = np.reshape( self.kernel(dx), (nSample, nPred) , order='F' )
        sy = F @ self.fitPar['beta'] + (self.fitPar['gamma'] @ r).T
        
        predictY=self.__Y_inverse_transform__(sy)

        rt=lstsq(self.fitPar['C'], r)[0]
        u = lstsq(self.fitPar['G'],
                             self.fitPar['Ft'].T @ rt - F.T)[0]
        
        mse = self.fitPar['sigma2'] * (1 + np.sum(u**2, axis=0) - np.sum(rt ** 2, axis=0)).T
        
        if only_value:
            return predictY
        else:
            return predictY, mse.reshape(-1, 1)
    
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain=self.__check_and_scale__(xTrain, yTrain)
        
        self.kernel.initialize(xTrain.shape[1])
        
        if(self.fitMode=='likelihood'):
            self._fit_likelihood(xTrain, yTrain)
        elif(self.fitMode=='predictError'):
            self._fit_predict_error(xTrain, yTrain)
            
        self.xTrain=xTrain; self.yTrain=yTrain
###-------------------private functions----------------------###
    def setKernel(self, kernel):
        
        self.kernel=kernel
        self.addSetting(kernel.setting)
        
    def _fit_predict_error(self, tol_xTrain, tol_yTrain):
        
        RS=RandSelect(20)
        train, test=RS.split(tol_xTrain)
        
        xTest=tol_xTrain[test,:]; yTest=tol_yTrain[test,:]
        xTrain=tol_xTrain[train,:]; yTrain=tol_yTrain[train,:]
        self.xTrain=xTrain; self.yTrain=yTrain
        
        F, D=self._initialize(xTrain)
        
        varList, varValue, ub, lb, position=self.setting.getParaInfos()
        nInput=len(varValue)
        
        if self.optimizer.type=="MP": 
            
            ###Using Mathematical Programming
            def objFunc(varValue):
                self.assignPara(position, varValue)
                self._objFunc(yTrain, F, D, record=True)
                yPred=self.predict(self.__X_inverse_transform__(xTest))
                obj=-1*r_square(self.__Y_inverse_transform__(yTest), yPred)
                
                return obj
            
            problem=PracticalProblem(objFunc, nInput, 1, ub, lb, x_labels=varList)
            res=self.optimizer.run(problem)
            bestDec=res.bestDec; bestObj=res.bestObj
            
            for _ in range(self.n_restart_optimize):
                res = self.optimizer(problem)
                obj = res.bestObj
                if obj < bestObj:
                    bestDec = res.bestDec
                    bestObj = obj
                    
        elif self.optimizer.type=="EA":
            ###Using Evolutionary Algorithm
            def objFunc(varValues):
                n, _=varValues.shape
                objs=np.zeros(n)
                
                for i, varValue in enumerate(varValues):
                    self.assignPara(position, varValue)
                    self._objFunc(yTrain, F, D, record=True)
                    yPred=self.predict(self.__X_inverse_transform__(xTest))
                    objs[i]=-1*r_square(self.__Y_inverse_transform__(yTest), yPred)
                return objs.reshape(-1, 1)
            
            problem=PracticalProblem(objFunc, nInput, 1, ub, lb, x_labels=varList)
            
            res=self.optimizer.run(problem)
            bestDec=res.bestDec; bestObj=res.bestObj
            
            for _ in range(self.n_restart_optimize):
                res=self.optimizer.run(problem)
                obj=res.bestObj
                if obj < bestObj:
                    bestDec = res.bestDec
                    bestObj = obj
                    
            self.assignPara(position, bestDec)
        
        self.xTrain=tol_xTrain; self.yTrain=tol_yTrain
        F, D=self._initialize(tol_xTrain)
        self._objFunc(self.yTrain, F, D, record=True)
        
    def _fit_likelihood(self, xTrain, yTrain):
                
        F, D=self._initialize(xTrain)  #fitPar
        
        varList, varValue, ub, lb, position=self.setting.getParaInfos()
        nInput=len(varValue)
        
        if self.optimizer.type=="MP":
            
            def objFunc(varValue):
                self.assignPara(position, varValue)
                return self._objFunc(yTrain, F, D, record=False)
            
            ###Using Mathematical Programming Method
            problem=PracticalProblem(objFunc, nInput, 1, ub, lb, x_labels=varList)
            
            res=self.optimizer.run(problem, xInit=np.array([1.0]))
            
            bestDec=res.bestDec; bestObj=res.bestObj
            
            for _ in range(self.n_restart_optimize):
                res = self.optimizer.run(problem)
                obj = res.bestObj
                if obj < bestObj:
                    bestDec = res.bestDec
                    bestObj = obj
                               
        elif self.optimizer.type=="EA":
            ###Using Evolutionary Algorithm
            def objFunc(thetas):

                objs=np.zeros(thetas.shape[0])
                for i, theta in enumerate(thetas):
                    self.assignPara(position, theta)
                    objs[i]=-1*self._objFunc(yTrain, F, D, record=False)
                    
                return objs.reshape(-1,1)
            
            problem=PracticalProblem(objFunc, nInput, 1, ub, lb, x_labels=varList)
            
            res=self.optimizer.run(problem)
            
            bestDec=res.bestDec; bestObj=res.bestObj
            
            for _ in range(self.n_restart_optimize):
                
                res=self.optimizer(problem)
                obj=res.bestObj
                if obj<bestObj:
                    bestDec=res.bestDec
                    bestObj=obj
        
        self.assignPara(position, bestDec)
        self._objFunc(yTrain, F, D, record=True)
        
    def _initialize(self, xTrain: np.ndarray):
        
        nSample, nFeature=xTrain.shape
        
        D = np.zeros((int((nSample*(nSample-1)/2)), nFeature))
        for k in range(nFeature):
            D[:, k] = pdist(xTrain[:, [k]], metric='euclidean')
        
        F=self.regrFunc(xTrain)
        
        self.fitPar={}
        
        return F, D
    
    def _objFunc(self, yTrain, F, D, record=False):
        
        obj=np.inf
        
        m=F.shape[0]
                
        r=self.kernel(D)
        
        mu = (10 + m) * np.spacing(1)
        R = np.triu(np.ones((m, m)), 1)
        R[R == 1.0] = r
        np.fill_diagonal(R, 1.0 + mu)
        try:
            C = cholesky(R).T
            Ft=lstsq(C, F)[0]
            Q, G = qr(Ft, mode='economic')
            
            Yt = lstsq(C, yTrain)[0]
            # Ytt = np.linalg.solve(C, yTrain)
            beta = lstsq(G, Q.T @ Yt)[0]
            rho = Yt - Ft @ beta
            sigma2 = np.sum(rho ** 2, axis=0) / m
            detR = np.prod(np.diag(C) ** (2 / m), axis=0)
            obj = np.sum(sigma2, axis=0) * detR
            
        except np.linalg.LinAlgError:
            return -np.inf
        
        if record:
            if isinstance(self.yScaler,  StandardScaler):
                self.fitPar['sigma2'] = np.square(self.yScaler.sita)@sigma2
            else:
                self.fitPar['sigma2'] = sigma2
            self.fitPar['beta'] = beta
            self.fitPar['gamma'] = (lstsq(C.T, rho)[0]).T
            self.fitPar['C'] = C
            self.fitPar['Ft'] = Ft
            self.fitPar['G'] = G.T
        
        return obj