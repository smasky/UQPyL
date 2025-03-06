import numpy as np
from scipy.linalg import cholesky, qr, lstsq
from scipy.spatial.distance import pdist
from typing import Literal, Tuple, Optional, Union


from .kernel import BaseKernel, Guass
from ..util.boxmin import Boxmin
from ..surrogateABC import Surrogate
from ...optimization.algorithmABC import Algorithm
from ...optimization.single_objective import GA
from ...utility.metrics import r_square
from ...utility.data_selections import RandSelect
from ...utility.scalers import Scaler, StandardScaler
from ...utility.polynomial_features import PolynomialFeatures
from ...problems import Problem

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
    A Kriging implementation based on python env. includes the new training method(prediction error), 
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
                * 'GA'
                *'Boxmin'
    
    nRes: the times of using evolutionary algorithms to optimize theta 
    
    fitMode: the objective function used to evaluate the performance of the theta, containing:
                *'likelihood' origin way
                *'predictError' new way
    
    normalized: the sign to normalize input data(x, y) or not
    
    Scale_type: the normalized method, containing:
            *'StandardScaler'
            *'MaxminScaler'
            
    """
    name = "KRG"
    
    def __init__(self, 
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                    polyFeature: PolynomialFeatures=None,
                        kernel: BaseKernel= Guass(),
                            regression: Literal['poly0','poly1','poly2']='poly0',
                                optimizer: Algorithm = "Boxmin",
                                nRestartTimes: int=1):
        
        super().__init__(scalers, polyFeature)

        self.kernel = None
        
        #set optimizer
        if optimizer == "Boxmin":
            self.optimizer = Boxmin()
        
        elif isinstance(optimizer, Algorithm):
            
            if optimizer.type=="EA":
                self.optimizer = optimizer
            else:
                print('The optimizer you input does not support! Here the GA would be used!')
                self.optimizer = GA(maxFEs=10000, nPop=50)
            
            self.optimizer.verboseFlag = False
            self.optimizer.saveFlag = False
            self.optimizer.logFlag = False
        
        else:
            print('The optimizer you input does not support! Here the Boxmin would be used!')
            self.optimizer = Boxmin()        
            
        #set the number of restart optimization
        self.nRes = nRestartTimes
        
        if not isinstance(kernel, BaseKernel):
            raise ValueError("The kernel must be the instance of surrogates.kriging.kernel!")
        
        self.setKernel(kernel)
        
        if(regression == 'poly0'):
            self.regrFunc = regrpoly0
            
        elif(regression == 'poly1'):
            self.regrFunc = regrpoly1
            
        elif(regression == 'poly2'):
            self.regrFunc = regrpoly2
        
###-------------------------------public function-----------------------------###
    def predict(self, xPred: np.ndarray, only_value=True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        xPred = self.__X_transform__(xPred)
        
        nSample, _ = self.xTrain.shape
        
        nPred, nFeature = xPred.shape
        
        dx = np.zeros( (nPred * nSample, nFeature) )
        
        kk = np.arange( nSample )
        
        for k in np.arange(nPred):
            dx[kk, :] = xPred[k, :] - self.xTrain
            kk = kk + nSample
        
        F = self.regrFunc(xPred)
        
        r = np.reshape( self.kernel(dx), (nSample, nPred) , order='F' )
        sy = F @ self.fitPar['beta'] + (self.fitPar['gamma'] @ r).T
        
        predictY = self.__Y_inverse_transform__(sy)

        rt = lstsq(self.fitPar['C'], r)[0]
        u = lstsq(self.fitPar['G'],
                             self.fitPar['Ft'].T @ rt - F.T)[0]
        
        mse = self.fitPar['sigma2'] * (1 + np.sum(u**2, axis=0) - np.sum(rt ** 2, axis=0)).T
        
        if only_value:
            return predictY
        else:
            return predictY, mse.reshape(-1, 1)
    
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain = self.__check_and_scale__(xTrain, yTrain)
        
        self.kernel.initialize(xTrain.shape[1])
        
        self._fit_likelihood(xTrain, yTrain)
        
###-------------------private functions----------------------###
    def setKernel(self, kernel):
        
        if self.kernel is not None:
            self.setting.removeSetting(self.kernel.setting) 
        
        self.kernel = kernel
        self.setting.mergeSetting(self.kernel.setting)
        self.kernel.setting = self.setting
    
    def _fitPure(self, xTrain, yTrain):
        
        self.xTrain = xTrain; self.yTrain = yTrain
        
        F, D= self._initialize(xTrain)
        
        self._objFunc(yTrain, F, D, record=True)
        
    def _fit_likelihood(self, xTrain, yTrain):
                
        F, D = self._initialize(xTrain)  #fitPar
        
        nameList = self.getParaList()
        
        paraInfos, ub, lb = self.setting.getParaInfos(nameList) #TODO
        
        nInput = ub.size
        
        if self.optimizer.type=="MP":
            
            def objFunc(varValue):
                self.setting.setVals(paraInfos, varValue)
                return self._objFunc(yTrain, F, D, record=False)
            
            ###Using Mathematical Programming Method
            problem = Problem(nInput, 1, ub, lb, objFunc = objFunc)
            
            bestDec , bestObj = self.optimizer.run(problem, xInit=np.repeat(np.array([1.0]), nInput))
              
            for _ in range(self.nRes):
                dec, obj = self.optimizer.run(problem)
                
                if obj < bestObj:
                    bestDec = dec
                    bestObj = obj
                               
        elif self.optimizer.type=="EA":
            ###Using Evolutionary Algorithm
            def objFunc(thetas):
                objs = np.zeros((thetas.shape[0], 1))
                thetas = np.exp(thetas)
                for i, theta in enumerate(thetas):
                    self.setting.setVals(paraInfos, theta)
                    objs[i, 0] = self._objFunc(yTrain, F, D, record=False)
                    
                return objs
            
            problem = Problem(nInput, 1, np.log(ub), np.log(lb), objFunc = objFunc)
            
            res = self.optimizer.run(problem)
            
            bestDec = np.exp(res.bestDec); bestObj=res.bestObj
            
            for _ in range(self.nRes):
                
                res = self.optimizer(problem)
                obj = res.bestObj
                if obj < bestObj:
                    bestDec = res.bestDec
                    bestObj = obj
        
        self.xTrain = xTrain; self.yTrain = yTrain
        self.setting.setVals(paraInfos, bestDec)
        self._objFunc(yTrain, F, D, record=True)
        
    def _initialize(self, xTrain: np.ndarray):
        
        nSample, nFeature = xTrain.shape
        
        D = np.zeros((int((nSample*(nSample-1)/2)), nFeature))
        for k in range(nFeature):
            D[:, k] = pdist(xTrain[:, [k]], metric='euclidean')
        
        F = self.regrFunc(xTrain)
        
        self.fitPar = {}
        
        return F, D
    
    def _objFunc(self, yTrain, F, D, record=False):
        
        obj = np.inf
        
        m = F.shape[0]
                
        r = self.kernel(D)
        
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
            return np.inf
        
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