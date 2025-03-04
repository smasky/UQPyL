from typing import Literal
import numpy as np

from .surrogateABC import Surrogate
from ..optimization.algorithmABC import Algorithm
from ..utility.model_selections import RandSelect
from ..utility.metrics import r_square
from ..problems.problem import Problem
class AutoTuner():
    def __init__(self, optimizer: Algorithm, model: Surrogate):
        
        self.optimizer = optimizer

        self.model = model
    
    def logIdx(self, paraInfos):
        
        parasType = self.model.setting.parasType
        
        I = []
        for name, idx in paraInfos.items():
            if parasType[name] == 0:
                I.append(idx)
        
        return np.concatenate(I)
           
    def opTune(self, xData: np.ndarray , yData: np.ndarray, paraList: list, ratio: int = 10, useLog: bool = True):
        
        xData, yData = self.model.__check_and_scale__(xData, yData)
        
        xDataCopy, yDataCopy = np.copy(xData), np.copy(yData) 
        
        if self.model.name in ["GPR", "KRG", "RBF"]:
            self.model.kernel.initialize(xData.shape[1])
        
        selector = RandSelect(ratio)
        
        trainIdx, testIdx = selector.split(xData)
        
        xTrain, yTrain = xData[trainIdx], yData[trainIdx]
        xTest, yTest = xData[testIdx], yData[testIdx]
        
        paraInfos, ub, lb, varType, varSet = self.model.setting.getParaInfos(paraList)
        nInput = ub.size
        
        if useLog:
            idx = self.logIdx(paraInfos)
        
        if self.optimizer.type == 'EA':
            
            def objFunc(X):
                
                Y = np.zeros((X.shape[0], 1))
                
                XX = X.copy()
                if useLog:
                    XX[:, idx] = np.exp(XX[:, idx])
                
                for i, x in enumerate(XX):
                    
                    self.model.setting.assignValues(paraInfos, x)
                    
                    try:
                        self.model._fitPure(xTrain, yTrain)
                            
                        yPred = self.model.predict(self.model.__X_inverse_transform__(xTest))
                            
                        obj = -1*r_square(self.model.__Y_inverse_transform__(yTest), yPred)
                    
                    except Exception:
                        obj = np.inf
                    
                    Y[i, 0] = obj
                    
                return Y
            
            if useLog:
                ub[idx] = np.log(ub[idx])
                lb[idx] = np.log(lb[idx])
            
            problem = Problem(nInput = nInput, nOutput = 1, ub = ub, lb = lb, 
                                objFunc = objFunc, varType = varType, varSet = varSet)
            
            res = self.optimizer.run(problem=problem)
            
            bestDecs = res.bestDecs.ravel(); bestObj = res.bestObjs.ravel()
            
            if useLog:
                bestDecs[idx] = np.exp(bestDecs[idx])
            
            self.model.setting.assignValues(paraInfos, bestDecs)
            
            self.model._fitPure(xDataCopy, yDataCopy)
            
            return bestDecs, bestObj
            
    def getParaList(self):
        
        return list(self.model.setting.parasValue.keys())