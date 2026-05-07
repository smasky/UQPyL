import abc
import numpy as np
from typing import Literal, Tuple

from .setting import Setting
from .scaler import Scaler

Scale_T=Tuple[Literal['StandardScaler','MinMaxScaler'], Literal['StandardScaler','MinMaxScaler']]

class SurrogateABC(metaclass = abc.ABCMeta):
    """
    Base class for surrogate models.

    This class defines the shared training and prediction workflow used by
    surrogate models in UQPyL, including:
    - input/output scaling
    - optional polynomial feature expansion
    - fitted-state management
    - optional uncertainty-output flag normalization

    Subclasses are expected to implement `fitModel`, and may override
    `fitHyper` when model-internal hyper-parameter optimization is needed.
    """
    supportsUncertainty = False

    def __init__(self, scalers = (None, None), polyFeature = None):
        
        #create user-define setting
        self.setting = Setting()
        self.setting.defaultOwner = "model"
        self.rng = np.random.default_rng()
        
        self.xScaler = scalers[0] if scalers[0] else None
        self.yScaler = scalers[1] if scalers[1] else None
        self.polyFeature = polyFeature if polyFeature else None

        self._parameterAppliers = {}
        
        self.xTrain = None
        self.yTrain = None
        self.fitState = {}

    def _prepare_training_components(self, xTrain: np.ndarray):
        '''
            Hook for model-specific prepared-data initialization,
            such as kernel initialization based on input dimension.
        '''
        return None
    
    def __check_and_scale__(self, xTrain: np.ndarray, yTrain: np.ndarray):
        '''
            check the type of train data
                and normalize the train data if required 
        '''
        
        if(not isinstance(xTrain,np.ndarray) or not isinstance(yTrain, np.ndarray)):
            raise ValueError('Please make sure the type of train_data is np.ndarry')

        xTrain = np.asarray(xTrain)
        yTrain = np.asarray(yTrain)

        if xTrain.ndim == 1:
            xTrain = xTrain.reshape(-1, 1)
        elif xTrain.ndim != 2:
            raise ValueError("xTrain must be a 1D or 2D array.")

        if yTrain.ndim == 1:
            yTrain = yTrain.reshape(-1, 1)
        elif yTrain.ndim != 2:
            raise ValueError("yTrain must be a 1D or 2D array.")
        
        if(xTrain.shape[0]==yTrain.shape[0]):
            
            xTrain = self.xScaler.fit_transform(xTrain) if self.xScaler else np.copy(xTrain)
            
            yTrain = self.yScaler.fit_transform(yTrain) if self.yScaler else np.copy(yTrain)
            
            xTrain = self.polyFeature.transform(xTrain) if self.polyFeature else np.copy(xTrain)
            
            return xTrain,yTrain
        
        else:
            
            raise ValueError("The shapes of x and y are not consistent. Please check them!")
    
    def __X_transform__(self,X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = self.xScaler.transform(X) if self.xScaler else X
        
        X = self.polyFeature.transform(X) if self.polyFeature else X
        
        return X
    
    def __Y_transform__(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        Y = self.yScaler.transform(Y) if self.yScaler else Y
        
        return Y
    
    def __Y_inverse_transform__(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        Y = self.yScaler.inverse_transform(Y) if self.yScaler else Y
            
        return Y
    
    def __X_inverse_transform__(self, X: np.ndarray) -> np.ndarray:
          
        X = self.xScaler.inverse_transform(X) if self.xScaler else X
        
        return X
    
    def getParaList(self):
        
        return list(self.setting.parVal.keys())

    def registerParameterApplier(self, name: str, applier):
        self._parameterAppliers[name] = applier
        return self

    def registerChoiceParameter(self, name: str, choices, owner: str = None):
        if not choices:
            raise ValueError(f"Choices for parameter '{name}' cannot be empty.")

        defaultChoice = choices[0]
        attr = {
            "lb": 0.0,
            "ub": float(len(choices)),
            "type": "choice",
            "set": list(choices),
            "log": False,
        }
        self.setting.set(name, defaultChoice, attr=attr, owner=owner)
        return self

    def isParameterActive(self, name: str):
        if name in self._parameterAppliers:
            return True

        return name in self.setting.parVal

    def applyParameterValues(self, paraList, values, ignoreInactive: bool = True):
        '''
            Apply a candidate parameter set under the current model context.
            Kernel-like structural parameters are applied first, then active tunable parameters.
        '''
        handledNames = []
        handledValues = []
        otherNames = []
        otherValues = []

        for idx, name in enumerate(paraList):
            value = values[idx]

            if self.setting.isChoicePara(name):
                value = self.setting.decodeValue(name, value)

            if name in self._parameterAppliers:
                handledNames.append(name)
                handledValues.append(value)
                continue

            if name not in self.setting.parVal:
                if ignoreInactive:
                    continue
                raise KeyError(f"Parameter '{name}' is not active for {self.__class__.__name__}.")

            otherNames.append(name)
            otherValues.append(value)

        for name, value in zip(handledNames, handledValues):
            self._parameterAppliers[name](value)

        activeNames = []
        activeValues = []
        for name, value in zip(otherNames, otherValues):
            if name not in self.setting.parVal:
                if ignoreInactive:
                    continue
                raise KeyError(f"Parameter '{name}' is not active for {self.__class__.__name__}.")

            activeNames.append(name)
            activeValues.append(value)

        if activeNames:
            paraInfos, _, _ = self.setting.getParaInfos(activeNames)
            self.setting.setVals(paraInfos, np.asarray(activeValues))

        return self

    def getParameterValues(self, *args, ignoreInactive: bool = False):
        values = []

        for name in args:
            if name in self._parameterAppliers:
                values.append(getattr(self, name))
                continue

            if name not in self.setting.parVal and name not in self.setting.parCon:
                if ignoreInactive:
                    values.append(None)
                    continue
                raise KeyError(f"Parameter '{name}' is not active for {self.__class__.__name__}.")

            values.append(self.setting.get(name))

        if len(args) > 1:
            return tuple(values)

        return values[0]

    def prepareTrainingData(self, xTrain: np.ndarray, yTrain: np.ndarray):
        '''
            Prepare raw training data into the canonical prepared-data form.
        '''
        xTrain, yTrain = self.__check_and_scale__(xTrain, yTrain)
        self._prepare_training_components(xTrain)
        return xTrain, yTrain

    def storeTrainingData(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self.xTrain = xTrain
        self.yTrain = yTrain
        return self

    def resetFitState(self):
        self.fitState = {}
        return self

    def requireFitted(self, *stateKeys):
        if self.xTrain is None or self.yTrain is None:
            raise RuntimeError(f"{self.__class__.__name__} has not been fitted yet.")

        missing = [key for key in stateKeys if key not in self.fitState]
        if missing:
            raise RuntimeError(
                f"{self.__class__.__name__} is missing fitted state: {', '.join(missing)}."
            )

        return self

    @abc.abstractmethod
    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        '''
            Fit the model under the current parameter setting using prepared data.
        '''
        pass

    def fitHyper(self, xTrain: np.ndarray, yTrain: np.ndarray):
        '''
            Run model-internal hyper-parameter optimization on prepared data.
            The default implementation directly fits the model under current parameters.
        '''
        return self.fitModel(xTrain, yTrain)

    def _normalize_predict_flags(self, returnStd: bool = False, returnVar: bool = False):
        """
        Normalize uncertainty request flags for predict().
        """
        if returnStd and returnVar:
            raise ValueError("Only one of returnStd and returnVar can be True.")

        if (returnStd or returnVar) and not self.supportsUncertainty:
            raise NotImplementedError(f"{self.__class__.__name__} does not support uncertainty output.")

        return returnStd, returnVar

    def _format_uncertainty_output(self, mean: np.ndarray, var: np.ndarray,
                                   returnStd: bool = False, returnVar: bool = False):
        """
        Format predict() outputs under the unified mean + std/var protocol.
        """
        returnStd, returnVar = self._normalize_predict_flags(returnStd, returnVar)

        mean = np.asarray(mean)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)

        if not (returnStd or returnVar):
            return mean

        var = np.asarray(var)
        if var.ndim == 1:
            var = var.reshape(-1, 1)

        if returnStd:
            return mean, np.sqrt(np.maximum(var, 0.0))

        return mean, var
         
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        xTrain, yTrain = self.prepareTrainingData(xTrain, yTrain)
        self.fitHyper(xTrain, yTrain)
        return self
    
    @abc.abstractmethod
    def predict(self, xPred: np.ndarray, returnStd: bool = False, returnVar: bool = False):
        pass
    
class MultiSurrogate():
    
    def __init__(self, n_surrogates, models_list=[]):
        self.n_surrogates=n_surrogates
        self.rng = np.random.default_rng()
        
        for model in models_list:
            if not isinstance(model, SurrogateABC):
                raise ValueError("Please append the type of surrogate!") 
                         
        self.models_list=models_list
        if self.models_list and len(self.models_list) != self.n_surrogates:
            raise ValueError("The number of surrogate models must match n_surrogates.")
        
    def append(self, model):
              
        if not isinstance(model, SurrogateABC):
            raise ValueError("Please append the type of surrogate!")
            
        self.models_list.append(model)
    
    def fit(self, trainX: np.ndarray, trainY: np.ndarray):
        trainY = np.asarray(trainY)
        if trainY.ndim == 1:
            trainY = trainY.reshape(-1, 1)

        if trainY.shape[1] != self.n_surrogates:
            raise ValueError("The number of outputs in trainY must match n_surrogates.")

        if len(self.models_list) != self.n_surrogates:
            raise ValueError("The number of models in models_list must match n_surrogates.")
        
        for i, model in enumerate(self.models_list):
            
            model.fit(trainX, trainY[:, i])
    
    def predict(self, testX: np.ndarray) -> np.ndarray:
        if len(self.models_list) != self.n_surrogates:
            raise ValueError("The number of models in models_list must match n_surrogates.")

        res=[]
        
        for model in self.models_list:
            pred = np.asarray(model.predict(testX))
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            res.append(pred)
            
        pre_Y=np.hstack(res)
        
        return pre_Y
