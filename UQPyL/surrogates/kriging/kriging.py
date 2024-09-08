import numpy as np
from scipy.linalg import LinAlgError, cholesky, qr, lstsq
from scipy.spatial.distance import pdist
from typing import Literal, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..surrogate_ABC import Surrogate, Scale_T
from .kernels import Krg_Kernel
from ...utility.metrics import r_square
from ...optimization import Boxmin, GA, MP_List, EA_List
from ...utility.model_selections import RandSelect
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures
from ...problems import ProblemABC as Problem

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

###-------------------------kernerls---------------------------###
# def guass(theta, d):
        
#     td = d * -theta
#     r = np.exp(np.sum(d * td, axis=1))
    
#     return r

# def exp(theta, d):
#     td= -theta
#     r= np.exp(np.sum(d*td, axis=1))
    
#     return r
# def cubic(theta, d):
#     td=np.sum(d*theta, axis=1)
#     ones=np.ones(td.shape)
#     td=np.minimum(ones, td)
#     r=1-3*td**2+2*td**3
#     return r

# K={"GAUSSIAN": guass, "EXP": exp, "CUBIC": cubic}

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
                *'GA' Genetic Algorithm
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
    def __init__(self, kernel: Krg_Kernel,
                 regression: Literal['poly0','poly1','poly2']='poly0',
                 optimizer: Literal['Boxmin', 'GA'] = 'Boxmin', 
                 fitMode: Literal['likelihood', 'predictError']='likelihood',
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 poly_feature: PolynomialFeatures=None,
                 n_restart_optimize: int=1):
        
        self.regression=regression
        self.optimizer=optimizer
        self.fitMode=fitMode
        self.n_restart_optimize=n_restart_optimize
        self.OPFunc=None
        
        if not isinstance(kernel, Krg_Kernel):
            raise ValueError("The kernel must be the instance of Krg_Kernel")
        
        self.kernel=kernel
        
        if(regression=='poly0'):
            self.regrFunc=regrpoly0
        elif(regression=='poly1'):
            self.regrFunc=regrpoly1
        elif(regression=='poly2'):
            self.regrFunc=regrpoly2
        
        super().__init__(scalers, poly_feature)
        
###-------------------------------public function-----------------------------###

    def predict(self,predictX: np.ndarray, only_value=True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        predictX=self.__X_transform__(predictX)
        n_sample, n_feature=self.trainX.shape
        n_pre,_=predictX.shape
        
        
        
        #TODO
        dx = np.zeros((n_pre * n_sample, n_feature))
        kk = np.arange(n_sample)

        for k in np.arange(n_pre):
            dx[kk, :] = predictX[k, :] - self.trainX
            kk = kk + n_sample
        ####################
        
        F=self.regrFunc(predictX)
        # self.kernel.theta=self.fitPar['theta'] #TODO
        r = np.reshape(self.kernel(dx),(n_sample, n_pre), order='F')
        sy = F @ self.fitPar['beta'] + (self.fitPar['gamma'] @ r).T
        
        predictY=self.__Y_inverse_transform__(sy)

        #mse
        rt=lstsq(self.fitPar['C'], r)[0]
        u = lstsq(self.fitPar['G'],
                             self.fitPar['Ft'].T @ rt - F.T)[0]
        mse = self.fitPar['sigma2'] * (1 + np.sum(u**2, axis=0) - np.sum(rt ** 2, axis=0)).T
        
        if only_value:
            return predictY
        else:
            return predictY, mse
    
    def fit(self, trainX: np.ndarray, trainY: np.ndarray):
        
        self.trainX, self.trainY=self.__check_and_scale__(trainX, trainY)
        
        _, n_feature=self.trainX.shape
        self.kernel.n_input=n_feature
        self.theta0=self.kernel.theta
        
        if(self.fitMode=='likelihood'):
            self._fit_likelihood()
        elif(self.fitMode=='predictError'):
            self._fit_predict_error()
            
###-------------------private functions----------------------###

    def _fit_predict_error(self):
        
        TotalX=self.trainX
        TotalY=self.trainY
        
        RS=RandSelect(20)
        train, test=RS.split(TotalX)
        
        testX=TotalX[test,:];testY=TotalY[test,:]
        trainX=TotalX[train,:];trainY=TotalY[train,:]
        self.F, self.D=self._initialize(trainX)
        
        if self.optimizer in MP_List: 
            ###Using Mathematical Programming
            self.OPModel=eval(self.optimizer)(self.theta0, self.kernel.theta_ub, self.kernel.theta_lb)

            #Wrap _objFunc
            if not self.OPFunc:
                def objFunc(theta):
                    self.trainY=trainY
                    self.trainX=trainX
                    self._objFunc(theta,record=True)
                    self.kernel.theta=theta
                    predictY=self.predict(self.__X_inverse_transform__(testX))
                    obj=-1*r_square(self.__Y_inverse_transform__(testY),predictY)
                    return obj
                
                self.OPFunc=objFunc
            bestTheta, obj=self.OPModel.run(self.OPFunc)
            self.kernel.theta=bestTheta
            
        elif self.optimizer in EA_List:
            ###Using Evolutionary Algorithm
            if not self.OPFunc:
                def objFunc(thetas):
                    self.trainY=trainY
                    self.trainX=trainX
                    objs=np.zeros(thetas.shape[0])
                    for i,theta in enumerate(thetas):
                        self._objFunc(np.power(np.e,theta),record=True)
                        self.kernel.theta=np.power(np.e,theta).ravel()
                        #TODO
                        predictY=self.predict(self.__X_inverse_transform__(testX))
                        objs[i]=-1*r_square(self.__Y_inverse_transform__(testY),predictY)
                    return objs.reshape(-1,1)
                self.OPFunc=objFunc
                
            problem=Problem(self.OPFunc, self.theta0.size, 1, np.log(self.kernel.theta_ub), np.log(self.kernel.theta_lb))
            
            self.OPModel=eval(self.optimizer)(problem, 50)

            bestObj=np.inf
            bestTheta=None
            for _ in range(self.n_restart_optimize):
                theta, obj=self.OPModel.run()
                if obj<bestObj:
                    bestTheta=theta
                    bestObj=obj
            self.kernel.theta=np.power(np.e,bestTheta).ravel()
        #reset
        self.OPFunc=None
        self.trainY=TotalY
        self.trainX=TotalX
        self.F, self.D=self._initialize(self.trainX)
        self._objFunc(self.kernel.theta,record=True)
        
    def _fit_likelihood(self):
        
        trainX=self.trainX
        trainY=self.trainY
        self.F, self.D=self._initialize(trainX)
        
        if self.optimizer in MP_List:
            ###Using Mathematical Programming
            self.OPModel=eval(self.optimizer)(self.theta0, self.kernel.theta_ub, self.kernel.theta_lb)
            bestTheta, _ =self.OPModel.run(self._objFunc)
            self.kernel.theta=bestTheta
            
        elif self.optimizer in EA_List:
            ###Using Evolutionary Algorithm
            def objFunc(thetas):
                self.trainY=trainY
                self.trainX=trainX
                objs=np.zeros(thetas.shape[0])
                for i,theta in enumerate(thetas):
                    objs[i]=self._objFunc(np.power(np.e,theta), record=False)
                return objs.reshape(-1,1)
            problem=Problem(objFunc, self.theta0.size, 1, np.log(self.kernel.theta_ub), np.log(self.kernel.theta_lb), )
            self.OPModel=eval(self.optimizer)(problem, 50)
            
            for _ in range(self.n_restart_optimize):
                bestObj=np.inf
                bestTheta=None
                theta, obj=self.OPModel.run()
                if obj<bestObj:
                    bestTheta=theta
                    bestObj=obj
                    
            self.kernel.theta=np.power(np.e,bestTheta)
        # Record coef
        self._objFunc(self.kernel.theta,record=True)
        
    def _initialize(self, trainX: np.ndarray):
        
        n_sample, n_feature=trainX.shape
        
        D = np.zeros((int((n_sample*(n_sample-1)/2)), n_feature))
        for k in range(n_feature):
            D[:, k] = pdist(trainX[:, [k]], metric='euclidean')
        
        F=self.regrFunc(trainX)
        
        self.fitPar={}
        
        return F, D
    
    def _objFunc(self,theta,record=False):
        
        obj=np.inf
        
        m=self.F.shape[0]
        #set theta to kernel
        self.kernel.theta=theta
        
        r=self.kernel(self.D)
        
        mu = (10 + m) * np.spacing(1)
        R = np.triu(np.ones((m, m)), 1)
        R[R == 1.0] = r
        np.fill_diagonal(R, 1.0 + mu)
        try:
            C = cholesky(R).T
            Ft=lstsq(C, self.F)[0]
            Q, G = qr(Ft, mode='economic')
            
            Yt = lstsq(C, self.trainY)[0]
            beta = lstsq(G, Q.T @ Yt)[0]
            rho = Yt - Ft @ beta
            sigma2 = np.sum(rho ** 2, axis=0) / m
            detR = np.prod(np.diag(C) ** (2 / m), axis=0)
            obj = np.sum(sigma2, axis=0) * detR
            
        except np.linalg.LinAlgError:
            return -np.inf
        
        
        if record:
            self.fitPar['sigma2'] = sigma2
            self.fitPar['beta'] = beta
            self.fitPar['gamma'] = (lstsq(C.T, rho)[0]).T
            self.fitPar['C'] = C
            self.fitPar['Ft'] = Ft
            self.fitPar['G'] = G.T
        
        return obj