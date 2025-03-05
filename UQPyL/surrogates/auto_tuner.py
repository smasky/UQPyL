import numpy as np

from .surrogateABC import Surrogate
from ..optimization.algorithmABC import Algorithm
from ..utility.data_selections import RandSelect
from ..utility.metrics import r_square
from ..problems.problem import Problem
class AutoTuner():
    def __init__(self, optimizer: Algorithm, model: Surrogate):
        
        self.optimizer = optimizer

        self.model = model
           
    def opTune(self, xData: np.ndarray , yData: np.ndarray, paraList: list, ratio: int = 10):
        
        xData, yData = self.model.__check_and_scale__(xData, yData)
        
        xDataCopy, yDataCopy = np.copy(xData), np.copy(yData) 
        
        # Initialize the kernel
        if self.model.name in ["GPR", "KRG", "RBF"]:
            self.model.kernel.initialize(xData.shape[1])
        
        selector = RandSelect(ratio)
        
        trainIdx, testIdx = selector.split(xData)
        
        xTrain, yTrain = xData[trainIdx], yData[trainIdx]
        xTest, yTest = xData[testIdx], yData[testIdx]
        
        paraInfos, ub, lb = self.model.setting.getParaInfos(paraList)
        nInput = ub.size
            
        def objFunc(X):
            
            Y = np.zeros((X.shape[0], 1))
            
            XX = X.copy()
            
            for i, x in enumerate(XX):
                
                self.model.setting.setVals(paraInfos, x)
                
                try:
                    self.model._fitPure(xTrain, yTrain)
                        
                    yPred = self.model.predict(self.model.__X_inverse_transform__(xTest))
                        
                    obj = -1*r_square(self.model.__Y_inverse_transform__(yTest), yPred)
                
                except Exception:
                    obj = np.inf
                
                Y[i, 0] = obj
                
            return Y
        
        problem = Problem(nInput = nInput, nOutput = 1, ub = ub, lb = lb, 
                            objFunc = objFunc)
        
        res = self.optimizer.run(problem=problem)
        
        bestDecs = res.bestDecs.ravel(); bestObj = res.bestObjs.ravel()
        
        self.model.setting.setVals(paraInfos, bestDecs)
        
        self.model._fitPure(xDataCopy, yDataCopy)
        
        return self.model.setting.getVals(*paraList), bestObj
    
    def getParaList(self):
        
        return list(self.model.setting.parasValue.keys())