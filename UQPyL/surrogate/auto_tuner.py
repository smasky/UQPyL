import numpy as np
from itertools import product

from .base import SurrogateABC
from ..optimization.base import AlgorithmABC
from .split import RandSelect, _resolveRng
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
        self.lastSplit = None
        self.candidateFailures = []
        self._candidateCount = 0
        self._modelSeed = None

    def _splitData(self, xRaw, ratio, seed, rng):
        self.candidateFailures = []
        self._candidateCount = 0
        parent = self.rng if seed is None and rng is None else _resolveRng(seed, rng)
        splitSeed, modelSeed, optimizerSeed = (spawn_seed(parent) for _ in range(3))
        trainIdx, testIdx = RandSelect(ratio).split(xRaw, seed=splitSeed)
        self._modelSeed = modelSeed
        self.lastSplit = {
            "seed": None if seed is None else int(seed),
            "split_seed": splitSeed, "model_seed": modelSeed, "optimizer_seed": optimizerSeed,
            "train_indices": trainIdx.copy(), "test_indices": testIdx.copy(),
        }
        return trainIdx, testIdx

    def _validateValidation(self, values):
        if len(values) < 2:
            raise ValueError('R2 validation requires at least two samples; increase ratio or supply more data.')
        if not np.all(np.isfinite(values)):
            raise ValueError('R2 validation outputs must be finite.')
        variation = np.sum((values - np.mean(values, axis=0))**2)
        if not np.isfinite(variation) or variation <= 0:
            raise ValueError('R2 validation requires nonconstant outputs with finite variation.')

    def _noValidCandidate(self):
        self.model.resetFitState()
        self.model.xTrain = self.model.yTrain = None
        raise RuntimeError('No candidate produced a finite validation score.')

    def _initialize_model_components(self, xData: np.ndarray):
        kernel = getattr(self.model, "kernel", None)
        if kernel is not None and hasattr(kernel, "initialize"):
            kernel.initialize(xData.shape[1])

    def _fit_with_mode(self, xTrain, yTrain, tuneMode):
        # Give candidate fits a reproducible random stream independent of search.
        if self._modelSeed is not None:
            self.model.rng = np.random.default_rng(self._modelSeed)
        if tuneMode == "joint":
            self.model.fitModel(xTrain, yTrain)
        elif tuneMode == "separate":
            self.model.fitHyper(xTrain, yTrain)
        else:
            raise ValueError("tuneMode must be either 'joint' or 'separate'.")

    def _scoreCandidate(self, xTrain, yTrain, xTest, yTest, tuneMode):
        candidateIndex = self._candidateCount
        self._candidateCount += 1
        try:
            self._fit_with_mode(xTrain, yTrain, tuneMode)
            prediction = self.model.predict(xTest)
            if not np.all(np.isfinite(prediction)):
                raise FloatingPointError('Candidate predictions are not finite.')
            score = r_square(yTest, prediction)
            if not np.isfinite(score):
                raise FloatingPointError('Candidate validation score is not finite.')
            return score
        except (np.linalg.LinAlgError, ArithmeticError) as error:
            self.candidateFailures.append({'candidate_index': candidateIndex,
                                           'error_type': type(error).__name__,
                                           'message': str(error)})
            return -np.inf

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
                tuneMode: str = "separate", *, seed=None, rng=None):
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

        # Fit preprocessing on the training split only; predict handles the
        # held-out raw inputs using those fitted components.
        trainIdx, testIdx = self._splitData(xRaw, ratio, seed, rng)
        self._validateValidation(yRaw[testIdx])
        xTrain, yTrain = self.model.prepareTrainingData(xRaw[trainIdx], yRaw[trainIdx])
        self.model.storeTrainingData(xTrain, yTrain)
        xTestRaw, yTestRaw = xRaw[testIdx], yRaw[testIdx]
        self._initialize_model_components(xTrain)
        paraList = self._resolve_para_list(paraList=paraList, owner=owner)
        
        paraInfos, ub, lb = self.model.setting.getParaInfos(paraList)
        nInput = ub.size
        encodings = {name: (self.model.setting.parType[name], self.model.setting.parLog[name])
                     for name in paraList}

        def applyCandidate(values):
            self.model.applyParameterValues(paraList, values, paraInfos=paraInfos)
            # An optimizer has one fixed box; switching kernels must not
            # silently reinterpret its coordinates or parameter bounds.
            for name, indices in paraInfos.items():
                setting = self.model.setting
                if name not in setting.parVal:
                    continue
                _, currentUpper, currentLower = setting.getParaInfos([name])
                if ((setting.parType[name], setting.parLog[name]) != encodings[name]
                        or not np.array_equal(currentUpper, ub[indices])
                        or not np.array_equal(currentLower, lb[indices])):
                    raise ValueError(f"Parameter '{name}' changed bounds or encoding during optTune; "
                                     "use compatible kernel settings or separate searches.")
            
        hasValidCandidate = False

        def objFunc(X):
            nonlocal hasValidCandidate
            
            Y = np.zeros((X.shape[0], 1))
            
            XX = X.copy()
            
            for i, x in enumerate(XX):
                
                applyCandidate(x)
                
                obj = self._scoreCandidate(xTrain, yTrain, xTestRaw, yTestRaw, tuneMode)
                hasValidCandidate = hasValidCandidate or np.isfinite(obj)

                Y[i, 0] = obj
                
            return Y
        
        problem = Problem(nInput = nInput, nObj = 1, ub = ub, lb = lb, 
                            objFunc = objFunc, optType = 'max')
        
        res = self.optimizer.run(problem=problem, seed=self.lastSplit["optimizer_seed"])
        bestTrueDecs = np.asarray(res.bestDecs).ravel()
        bestTrueObj = np.asarray(res.bestObjs).ravel()
        if not hasValidCandidate or not np.all(np.isfinite(bestTrueObj)):
            self._noValidCandidate()
        
        applyCandidate(bestTrueDecs)
        
        # Refit preprocessing and the selected model on all supplied data.
        xFull, yFull = self.model.prepareTrainingData(xRaw, yRaw)
        self._fit_with_mode(xFull, yFull, tuneMode)
        
        return self.model.getParameterValues(*paraList, ignoreInactive=True), bestTrueObj

    def _applyGridCandidate(self, paraList, candidate):
        # Grid entries are grouped by parameter, so a vector is one candidate.
        parts = [np.asarray(value, dtype=object).ravel() for value in candidate]
        offsets = np.cumsum([0] + [part.size for part in parts])
        paraInfos = {name: np.arange(offsets[i], offsets[i + 1]) for i, name in enumerate(paraList)}
        self.model.applyParameterValues(paraList, np.concatenate(parts), paraInfos=paraInfos)
    
    def gridTune(self, xData: np.ndarray, yData: np.ndarray, paraGrid: dict = None,
                 ratio: int = 10, owner: str = None,
                 tuneMode: str = "separate", *, seed=None, rng=None):
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

        # Fit preprocessing on the training split only; predict handles the
        # held-out raw inputs using those fitted components.
        trainIdx, testIdx = self._splitData(xRaw, ratio, seed, rng)
        self._validateValidation(yRaw[testIdx])
        xTrain, yTrain = self.model.prepareTrainingData(xRaw[trainIdx], yRaw[trainIdx])
        self.model.storeTrainingData(xTrain, yTrain)
        xTestRaw, yTestRaw = xRaw[testIdx], yRaw[testIdx]
        self._initialize_model_components(xTrain)

        if paraGrid is None:
            paraList = self._resolve_para_list(paraList=None, owner=owner)
            paraGrid = {}
            for name in paraList:
                value = self.model.setting.parVal[name].copy()
                # Match the encoded coordinates accepted by explicit grids.
                with np.errstate(divide="ignore"):
                    value = np.log(value) if self.model.setting.parLog[name] else value
                paraGrid[name] = [value]
        else:
            paraList = list(paraGrid.keys())

        choices = [list(items) for items in paraGrid.values()]
        if not choices or any(not items for items in choices):
            raise ValueError("paraGrid must contain at least one candidate per parameter.")
        paraCombs = product(*choices)
        
        #Grid search
        bestObj = -np.inf
        bestDecs = None
        
        for paraComb in paraCombs:
            
            self._applyGridCandidate(paraList, paraComb)
            
            obj = self._scoreCandidate(xTrain, yTrain, xTestRaw, yTestRaw, tuneMode)

            if obj > bestObj:
                
                bestObj = obj
                bestDecs = paraComb
                
        if bestDecs is None:
            self._noValidCandidate()
        self._applyGridCandidate(paraList, bestDecs)
        
        # Refit preprocessing and the selected model on all supplied data.
        xFull, yFull = self.model.prepareTrainingData(xRaw, yRaw)
        self._fit_with_mode(xFull, yFull, tuneMode)
        
        return self.model.getParameterValues(*paraList, ignoreInactive=True), bestObj
