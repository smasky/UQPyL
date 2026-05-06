import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from typing import Any, Tuple, Optional
from copy import deepcopy

from .kernel import BaseKernel, RBF
from ..util.boxmin import Boxmin
from ...problem import Problem
from ..base import SurrogateABC
from ...optimization import AlgorithmABC
from ...util.scaler import Scaler
from ...util.poly import PolyFeature

class GPR(SurrogateABC):
    
    name = "GPR"
    supportsUncertainty = True
    internalOptimizerFamilies = ("MP", "EA")
    internalMPOptimizer = "Boxmin"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                    polyFeature: PolyFeature = None,
                        kernel: BaseKernel = RBF(),
                            optimizer: AlgorithmABC = "Boxmin", nRestartTimes: int = 0,
                                    C: float = 1e-9,
                                    C_attr: dict = {'ub': 1e-6, 'lb':1e-12, 
                                                        'type': 'float', 
                                                        'log': 'True'}):
        
        super().__init__(scalers=scalers, polyFeature=polyFeature)
        
        self.kernel = None
        
        self.setting.setPara("C", C, C_attr)
        
        if optimizer == "Boxmin":
            optimizer = Boxmin()
        elif isinstance(optimizer, AlgorithmABC):
            alg_type = getattr(optimizer, "alg_type", getattr(optimizer, "type", None))
            optimizer.verboseFlag = False
            optimizer.saveFlag = False
            optimizer.logFlag = False
            if alg_type not in self.internalOptimizerFamilies:
                raise ValueError(
                    "GPR internal optimizer only supports MP (currently Boxmin) or EA."
                )
        else:
            raise ValueError(
                "GPR optimizer must be 'Boxmin' or an AlgorithmABC instance in MP/EA."
            )
            
        self.optimizer = optimizer

        self.registerParameterApplier("kernel", self.setKernel)
        self._kernelChoiceRegistered = False
        
        self.setKernel(kernel)
        
        self.nRes = nRestartTimes

    def _prepare_training_components(self, xTrain: np.ndarray):
        self.kernel.initialize(xTrain.shape[1])

    def _invalidate_fit_after_structure_change(self):
        self.resetFitState()
        return self
        
###---------------------------------public function---------------------------------------###
    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)
        self._objfunc(xTrain, yTrain, record=True)
        return self

    def fitHyper(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self._optimizeHyper(xTrain, yTrain)
        return self
            
    def predict(self, xPred: np.ndarray, returnStd: bool = False,
                returnVar: bool = False):
        returnStd, returnVar = self._normalize_predict_flags(returnStd, returnVar)
        self.requireFitted("L", "alpha")
        
        xPred = self.__X_transform__(xPred)
        
        K_trans = self.kernel(xPred, self.xTrain)
        y_mean = K_trans @ self.fitState["alpha"]
               
        V = solve_triangular(
            self.fitState["L"], K_trans.T, lower=True
        )
        
        K = self.kernel(xPred)
        y_var = np.diag(K).copy()
        y_var -= np.einsum("ij, ji->i", V.T, V)
        y_var[y_var<0] = 0.0

        return self._format_uncertainty_output(
            self.__Y_inverse_transform__(y_mean),
            y_var.reshape(-1, 1),
            returnStd=returnStd,
            returnVar=returnVar,
        )
    
###--------------------------private functions--------------------###    
    def _optimizeHyper(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
            Internal hyper-parameter optimization.

            Current supported optimizer families:
                - MP: currently implemented by Boxmin
                - EA: evolutionary algorithms with alg_type == "EA"
        """
        nameList = self.getParaList()
        
        paraInfos, ub, lb = self.setting.getParaInfos(nameList)
        
        nInput = ub.size
        
        alg_type = getattr(self.optimizer, "alg_type", getattr(self.optimizer, "type", None))
        if alg_type == "MP":
            
            def objFunc(varValue):

                self.setting.setVals(paraInfos, varValue)
                
                return self._objfunc(xTrain, yTrain, record = False)
                
            problem = Problem(nInput = nInput, nObj = 1, ub = ub, lb = lb, 
                                objFunc = objFunc)
            
            bestDecs, bestObj = self.optimizer.run(problem)
        
        elif alg_type == "EA":
            
            def objFunc(varValues):
                
                objs = np.zeros(varValues.shape[0])
                
                for i, value in enumerate(varValues):
                    
                    self.setting.setVals(paraInfos, value)
                    
                    objs[i] = self._objfunc(xTrain, yTrain, record=False)
                    
                return objs.reshape( (-1, 1) )
            
            problem = Problem(nInput, 1, ub, lb, objFunc = objFunc)
            
            res = self.optimizer.run(problem)
            bestDecs = np.asarray(res.bestDecs).ravel()
            bestObj = float(np.asarray(res.bestObjs).reshape(-1)[0])
            
            for _ in range(self.nRes):
                
                res = self.optimizer.run(problem)
                dec = np.asarray(res.bestDecs).ravel()
                obj = float(np.asarray(res.bestObjs).reshape(-1)[0])
                
                if obj < bestObj:
                    bestDecs, bestObj = dec, obj
        else:
            raise ValueError(
                "GPR internal optimizer only supports MP (currently Boxmin) or EA."
            )
                    
        self.setting.setVals(paraInfos, np.asarray(bestDecs).ravel())
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)
        self._objfunc(xTrain, yTrain, record=True)
        
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
            self.fitState["L"] = L
            self.fitState["alpha"] = alpha
            self.fitState["objective"] = log_likelihood

        return log_likelihood
    
    def setKernel(self, kernel: BaseKernel):
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
