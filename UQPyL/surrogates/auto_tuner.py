import numpy as np

from .surrogateABC import Surrogate
from ..optimization.algorithmABC import Algorithm
from ..utility.data_selections import RandSelect
from ..utility.metrics import r_square
from ..problems.problem import Problem
class AutoTuner():
    '''
    AutoTuner class
    '''
    def __init__(self, model: Surrogate, optimizer: Algorithm = None):
        '''
        Initialize the AutoTuner
        :param model: Surrogate, the surrogate model
        :param optimizer: Algorithm, the optimizer
        '''
        self.optimizer = optimizer

        self.model = model
           
    def optTune(self, xData: np.ndarray , yData: np.ndarray, paraList: list, ratio: int = 10):
        '''
        Optimize the hyper-parameters for the surrogate model
        :param xData: np.ndarray, the input data
        :param yData: np.ndarray, the output data
        :param paraList: list, the parameter list
        :param ratio: int, the ratio of the training data
        :return: tuple, the best parameter combination and the best objective value
        '''
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
                
                except Exception as e:
                    
                    print(f"Warning: Error in fitting the model: {e}")
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
    
    def gridTune(self, xData: np.ndarray, yData: np.ndarray, paraGrid: dict, ratio: int = 10):
        '''
        Grid search for the best parameter combination
        :param xData: np.ndarray, the input data
        :param yData: np.ndarray, the output data
        :param paraGrid: dict, the parameter grid
        :param ratio: int, the ratio of the training data
        :return: tuple, the best parameter combination and the best objective value
        '''
        xData, yData = self.model.__check_and_scale__(xData, yData)
        
        xDataCopy, yDataCopy = np.copy(xData), np.copy(yData)
        
        # TODO
        paraCombs = np.meshgrid(*paraGrid.values())
        paraCombs = np.array([arr.ravel() for arr in paraCombs]).T
        
        paraList = list(paraGrid.keys())
        
        selector = RandSelect(ratio)
        
        trainIdx, testIdx = selector.split(xData)
        
        xTrain, yTrain = xData[trainIdx], yData[trainIdx]
        xTest, yTest = xData[testIdx], yData[testIdx]
        
        paraInfos, _, _ = self.model.setting.getParaInfos(paraList)
        
        #Grid search
        bestObj = np.inf
        bestDecs = None
        
        for paraComb in paraCombs:
            
            self.model.setting.setVals(paraInfos, paraComb)
            
            try:
                self.model._fitPure(xTrain, yTrain)
                
                yPred = self.model.predict(self.model.__X_inverse_transform__(xTest))
                
                obj = -1*r_square(self.model.__Y_inverse_transform__(yTest), yPred)
            
            except Exception as e:
                
                print(f"Warning: Error in fitting the model: {e}")
                obj = np.inf
                
            if obj < bestObj:
                
                bestObj = obj
                bestDecs = paraComb
                
        self.model.setting.setVals(paraInfos, bestDecs)
        
        self.model._fitPure(xDataCopy, yDataCopy)
        
        return self.model.setting.getVals(*paraList), bestObj