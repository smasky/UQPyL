import numpy as np

from .base import SurrogateABC
from ..optimization.base import AlgorithmABC
from .split import RandSelect
from .metric import r_square
from ..problem.problem import Problem
from ..core import spawn_seed

class AutoTuner():
    '''
    Hyper-parameter tuner for surrogate models.

    The tuner evaluates candidate parameter settings by fitting the target
    surrogate on a train split and scoring predictions on a validation split.
    It supports both optimizer-driven tuning (`optTune`) and explicit grid
    search (`gridTune`).

    Examples:
        >>> tuner = AutoTuner(model, optimizer)
        >>> bestParams, bestScore = tuner.optTune(xData, yData)
    '''
    def __init__(self, model: SurrogateABC, optimizer: AlgorithmABC = None):
        '''
        Initialize the AutoTuner
        :param model: Surrogate, the surrogate model
        :param optimizer: Algorithm, the optimizer
        '''
        self.optimizer = optimizer

        self.model = model
        self.rng = np.random.default_rng()

    def _initialize_model_components(self, xData: np.ndarray):
        kernel = getattr(self.model, "kernel", None)
        if kernel is not None and hasattr(kernel, "initialize"):
            kernel.initialize(xData.shape[1])

    def _fit_with_mode(self, xTrain, yTrain, tuneMode):
        if tuneMode == "joint":
            self.model.fitModel(xTrain, yTrain)
        elif tuneMode == "separate":
            self.model.fitHyper(xTrain, yTrain)
        else:
            raise ValueError("tuneMode must be either 'joint' or 'separate'.")

    def _resolve_para_list(self, paraList = None, owner = None):
        if paraList is not None:
            return list(paraList)

        paraList = self.model.setting.getParaList(owner=owner, tunableOnly=True)
        if not paraList:
            ownerMsg = "" if owner is None else f" for owner '{owner}'"
            raise ValueError(f"No tunable parameters found{ownerMsg}.")

        return paraList
           
    def optTune(self, xData: np.ndarray , yData: np.ndarray, paraList: list = None,
                ratio: int = 10, owner: str = None,
                tuneMode: str = "separate"):
        '''
        Optimize the hyper-parameters for the surrogate model
        :param xData: np.ndarray, the input data
        :param yData: np.ndarray, the output data
        :param paraList: list, optional parameter names to tune
        :param ratio: int, the ratio of the training data
        :param owner: str, optional owner filter such as `model` or `kernel`
        :return: tuple, the best parameter combination and the best objective value
        '''
        xRaw = np.asarray(xData)
        yRaw = np.asarray(yData)
        if xRaw.ndim == 1:
            xRaw = xRaw.reshape(-1, 1)
        if yRaw.ndim == 1:
            yRaw = yRaw.reshape(-1, 1)

        xData, yData = self.model.prepareTrainingData(xRaw, yRaw)
        
        xDataCopy, yDataCopy = np.copy(xData), np.copy(yData) 

        self._initialize_model_components(xData)
        paraList = self._resolve_para_list(paraList=paraList, owner=owner)
        
        selector = RandSelect(ratio)
        
        trainIdx, testIdx = selector.split(xRaw)
        
        xTrain, yTrain = xData[trainIdx], yData[trainIdx]
        xTestRaw, yTestRaw = xRaw[testIdx], yRaw[testIdx]
        
        paraInfos, ub, lb = self.model.setting.getParaInfos(paraList)
        nInput = ub.size
            
        def objFunc(X):
            
            Y = np.zeros((X.shape[0], 1))
            
            XX = X.copy()
            
            for i, x in enumerate(XX):
                
                self.model.applyParameterValues(paraList, x, ignoreInactive=True)
                
                try:
                    self._fit_with_mode(xTrain, yTrain, tuneMode)
                        
                    yPred = self.model.predict(xTestRaw)
                        
                    obj = r_square(yTestRaw, yPred)
                
                except Exception as e:
                    
                    print(f"Warning: Error in fitting the model: {e}")
                    obj = -np.inf
                
                Y[i, 0] = obj
                
            return Y
        
        problem = Problem(nInput = nInput, nObj = 1, ub = ub, lb = lb, 
                            objFunc = objFunc, optType = 'max')
        
        res = self.optimizer.run(problem=problem, seed=spawn_seed(self.rng))
        bestTrueDecs = np.asarray(res.bestDecs).ravel()
        bestTrueObj = np.asarray(res.bestObjs).ravel()
        
        self.model.applyParameterValues(paraList, bestTrueDecs, ignoreInactive=True)
        
        self._fit_with_mode(xDataCopy, yDataCopy, tuneMode)
        
        return self.model.getParameterValues(*paraList), bestTrueObj
    
    def gridTune(self, xData: np.ndarray, yData: np.ndarray, paraGrid: dict = None,
                 ratio: int = 10, owner: str = None,
                 tuneMode: str = "separate"):
        '''
        Grid search for the best parameter combination
        :param xData: np.ndarray, the input data
        :param yData: np.ndarray, the output data
        :param paraGrid: dict, optional parameter grid
        :param ratio: int, the ratio of the training data
        :param owner: str, optional owner filter used when paraGrid is not provided
        :return: tuple, the best parameter combination and the best objective value
        '''
        xRaw = np.asarray(xData)
        yRaw = np.asarray(yData)
        if xRaw.ndim == 1:
            xRaw = xRaw.reshape(-1, 1)
        if yRaw.ndim == 1:
            yRaw = yRaw.reshape(-1, 1)

        xData, yData = self.model.prepareTrainingData(xRaw, yRaw)
        
        xDataCopy, yDataCopy = np.copy(xData), np.copy(yData)

        self._initialize_model_components(xData)

        if paraGrid is None:
            paraList = self._resolve_para_list(paraList=None, owner=owner)
            paraGrid = {
                name: np.asarray(self.model.getParameterValues(name)).reshape(-1).tolist()
                for name in paraList
            }
        else:
            paraList = list(paraGrid.keys())

        paraCombs = np.meshgrid(*paraGrid.values())
        paraCombs = np.array([arr.ravel() for arr in paraCombs]).T
        
        selector = RandSelect(ratio)
        
        trainIdx, testIdx = selector.split(xRaw)
        
        xTrain, yTrain = xData[trainIdx], yData[trainIdx]
        xTestRaw, yTestRaw = xRaw[testIdx], yRaw[testIdx]
        
        #Grid search
        bestObj = -np.inf
        bestDecs = None
        
        for paraComb in paraCombs:
            
            self.model.applyParameterValues(paraList, paraComb, ignoreInactive=True)
            
            try:
                self._fit_with_mode(xTrain, yTrain, tuneMode)
                
                yPred = self.model.predict(xTestRaw)
                
                obj = r_square(yTestRaw, yPred)
                # Guard against NaN/Inf (e.g., degenerate test split).
                if not np.isfinite(obj):
                    obj = -np.inf
            
            except Exception as e:
                
                print(f"Warning: Error in fitting the model: {e}")
                obj = -np.inf
                
            if obj > bestObj:
                
                bestObj = obj
                bestDecs = paraComb
                
        # If all candidates failed (or produced NaN), fall back to the first combination.
        if bestDecs is None:
            bestDecs = paraCombs[0]
        self.model.applyParameterValues(paraList, bestDecs, ignoreInactive=True)
        
        self._fit_with_mode(xDataCopy, yDataCopy, tuneMode)
        
        return self.model.getParameterValues(*paraList), bestObj
