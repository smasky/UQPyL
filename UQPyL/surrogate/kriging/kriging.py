import numpy as np
from scipy.linalg import cholesky, qr, lstsq
from scipy.spatial.distance import pdist
from typing import Literal, Tuple, Optional, Union
from copy import deepcopy


from .kernel import BaseKernel, Guass
from ..util.boxmin import Boxmin
from ..base import SurrogateABC
from ...optimization.base import AlgorithmABC
from ...optimization.soea import GA
from ...util.metric import r_square
from ...util.split import RandSelect
from ...util.scaler import Scaler, StandardScaler
from ...util.poly import PolyFeature
from ...problem import Problem

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

class KRG(SurrogateABC):
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
    
    optimizer: internal hyper optimizer for theta, supporting:
                * 'Boxmin'
                * EA algorithm objects
    
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
    supportsUncertainty = True
    internalOptimizerFamilies = ("MP", "EA")
    internalMPOptimizer = "Boxmin"
    
    def __init__(self, 
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                    polyFeature: PolyFeature=None,
                        kernel: BaseKernel= Guass(),
                            regression: Literal['poly0','poly1','poly2']='poly0',
                                optimizer: AlgorithmABC = "Boxmin",
                                nRestartTimes: int=1):
        
        super().__init__(scalers, polyFeature)

        self.kernel = None
        
        # set internal hyper optimizer:
        # - MP family: currently implemented by Boxmin
        # - EA family: evolutionary algorithms with alg_type == "EA"
        if optimizer == "Boxmin":
            self.optimizer = Boxmin()
        
        elif isinstance(optimizer, AlgorithmABC):
            alg_type = getattr(optimizer, "alg_type", getattr(optimizer, "type", None))
            optimizer.verboseFlag = False
            optimizer.saveFlag = False
            optimizer.logFlag = False

            if alg_type not in self.internalOptimizerFamilies:
                raise ValueError(
                    "KRG internal optimizer only supports MP (currently Boxmin) or EA."
                )

            self.optimizer = optimizer
        
        else:
            raise ValueError(
                "KRG optimizer must be 'Boxmin' or an AlgorithmABC instance in MP/EA."
            )
            
        #set the number of restart optimization
        self.nRes = nRestartTimes

        self.registerParameterApplier("kernel", self.setKernel)
        self._kernelChoiceRegistered = False
        
        if not isinstance(kernel, BaseKernel):
            raise ValueError("The kernel must be the instance of surrogates.kriging.kernel!")
        
        self.setKernel(kernel)
        
        if(regression == 'poly0'):
            self.regrFunc = regrpoly0
            
        elif(regression == 'poly1'):
            self.regrFunc = regrpoly1
            
        elif(regression == 'poly2'):
            self.regrFunc = regrpoly2

    def _prepare_training_components(self, xTrain: np.ndarray):
        self.kernel.initialize(xTrain.shape[1])

    def _invalidate_fit_after_structure_change(self):
        self.resetFitState()
        return self
        
###-------------------------------public function-----------------------------###
    def predict(self, xPred: np.ndarray, returnStd: bool = False,
                returnVar: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        returnStd, returnVar = self._normalize_predict_flags(returnStd, returnVar)
        self.requireFitted("beta", "gamma", "C", "Ft", "G", "sigma2")
        
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
        sy = F @ self.fitState['beta'] + (self.fitState['gamma'] @ r).T
        
        predictY = self.__Y_inverse_transform__(sy)

        rt = lstsq(self.fitState['C'], r)[0]
        u = lstsq(self.fitState['G'],
                             self.fitState['Ft'].T @ rt - F.T)[0]
        
        var = self.fitState['sigma2'] * (1 + np.sum(u**2, axis=0) - np.sum(rt ** 2, axis=0)).T

        return self._format_uncertainty_output(
            predictY,
            var.reshape(-1, 1),
            returnStd=returnStd,
            returnVar=returnVar,
        )
    
    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)
        F, D = self._initialize(xTrain)
        self._objFunc(yTrain, F, D, record=True)
        return self

    def fitHyper(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self._optimizeHyper(xTrain, yTrain)
        return self
        
###-------------------private functions----------------------###
    def setKernel(self, kernel):
        oldKernelNames = []
        if self.kernel is not None:
            oldKernelNames = [
                name for name in self.kernel.setting.getParaList(owner="kernel", tunableOnly=False)
                if name != "kernel"
            ]

        kernelChoiceValue = self.setting.parVal.get("kernel", None)
        kernelChoiceAttr = self.setting.parSet.get("kernel", None)
        kernelChoiceOwner = self.setting.parOwner.get("kernel", None)

        if oldKernelNames:
            self.setting.removeParas(oldKernelNames)

        if not hasattr(kernel, "_templateSetting"):
            kernel._templateSetting = deepcopy(kernel.setting)
        kernel.setting = deepcopy(kernel._templateSetting)
        
        self.kernel = kernel
        self.setting.mergeSetting(self.kernel.setting)
        self.kernel.setting = self.setting

        if kernelChoiceValue is not None and kernelChoiceAttr is not None:
            self.setting.parVal["kernel"] = self.setting._normalize_choice_array(kernel, kernelChoiceAttr)
            self.setting.parSet["kernel"] = kernelChoiceAttr
            self.setting.parType["kernel"] = 2
            self.setting.parOwner["kernel"] = kernelChoiceOwner
            self.setting.parLB["kernel"] = np.asarray([0.0])
            self.setting.parUB["kernel"] = np.asarray([float(len(kernelChoiceAttr[0]))])
            self.setting.parLog["kernel"] = False

        if self.xTrain is not None:
            self.kernel.initialize(self.xTrain.shape[1])
        self._invalidate_fit_after_structure_change()

    def setKernelChoices(self, kernels):
        self.registerChoiceParameter("kernel", kernels, owner="kernel")
        self._kernelChoiceRegistered = True
        return self
    
    def _optimizeHyper(self, xTrain, yTrain):
        """
            Internal hyper-parameter optimization.

            Current supported optimizer families:
                - MP: currently implemented by Boxmin
                - EA: evolutionary algorithms with alg_type == "EA"
        """
        F, D = self._initialize(xTrain)  #fitPar
        
        nameList = self.getParaList()
        
        paraInfos, ub, lb = self.setting.getParaInfos(nameList) #TODO
        
        nInput = ub.size
        
        alg_type = getattr(self.optimizer, "alg_type", getattr(self.optimizer, "type", None))
        if alg_type == "MP":
            
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
                               
        elif alg_type == "EA":
            ###Using Evolutionary Algorithm
            def objFunc(varValues):
                objs = np.zeros((varValues.shape[0], 1))
                for i, value in enumerate(varValues):
                    self.setting.setVals(paraInfos, value)
                    objs[i, 0] = self._objFunc(yTrain, F, D, record=False)
                    
                return objs
            
            problem = Problem(nInput, 1, ub, lb, objFunc = objFunc)
            
            res = self.optimizer.run(problem)
            
            bestDec = np.asarray(res.bestDecs).ravel()
            bestObj = float(np.asarray(res.bestObjs).reshape(-1)[0])
            
            for _ in range(self.nRes):
                
                res = self.optimizer.run(problem)
                obj = float(np.asarray(res.bestObjs).reshape(-1)[0])
                if obj < bestObj:
                    bestDec = np.asarray(res.bestDecs).ravel()
                    bestObj = obj
        else:
            raise ValueError(
                "KRG internal optimizer only supports MP (currently Boxmin) or EA."
            )
        
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)
        self.setting.setVals(paraInfos, bestDec)
        self._objFunc(yTrain, F, D, record=True)
        
    def _initialize(self, xTrain: np.ndarray):
        
        nSample, nFeature = xTrain.shape
        
        D = np.zeros((int((nSample*(nSample-1)/2)), nFeature))
        for k in range(nFeature):
            D[:, k] = pdist(xTrain[:, [k]], metric='euclidean')
        
        F = self.regrFunc(xTrain)
        
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
                self.fitState['sigma2'] = np.square(self.yScaler.sita)@sigma2
            else:
                self.fitState['sigma2'] = sigma2
            self.fitState['beta'] = beta
            self.fitState['gamma'] = (lstsq(C.T, rho)[0]).T
            self.fitState['C'] = C
            self.fitState['Ft'] = Ft
            self.fitState['G'] = G.T
            self.fitState['objective'] = obj

        return obj
