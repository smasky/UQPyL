import abc
from copy import deepcopy
import numpy as np
from typing import Literal, Tuple

from .setting import Setting
from .scaler import Scaler
from ..core import spawn_seed

Scale_T = Tuple[Literal["StandardScaler", "MinMaxScaler"], Literal["StandardScaler", "MinMaxScaler"]]


class SurrogateABC(metaclass=abc.ABCMeta):
    """
    Base class for surrogate models.

    This class defines the shared training and prediction workflow used by
    surrogate models in UQPyL, including:
    - input/output scaling
    - optional polynomial feature expansion
    - fitted-state management
    - optional uncertainty-output flag normalization

    Each model fits one output column. MultiSurrogate owns multiple outputs.

    Subclasses are expected to implement `fitModel`, and may override
    `fitHyper` when model-internal hyper-parameter optimization is needed.
    """

    supportsUncertainty = False

    def __init__(self, scalers=(None, None), polyFeature=None):

        # create user-define setting
        self.setting = Setting()
        self.setting.defaultOwner = "model"
        self.rng = np.random.default_rng()

        self.xScaler = deepcopy(scalers[0]) if scalers[0] is not None else None
        self.yScaler = deepcopy(scalers[1]) if scalers[1] is not None else None
        self.polyFeature = polyFeature if polyFeature else None

        self._parameterAppliers = {}

        self.xTrain = None
        self.yTrain = None
        self.fitState = {}
        self._rawInputCount = None

    def _prepare_training_components(self, xTrain: np.ndarray):
        """
        Hook for model-specific prepared-data initialization,
        such as kernel initialization based on input dimension.
        """
        return None

    def _checkAndScale(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        check the type of train data
            and normalize the train data if required
        """

        if not isinstance(xTrain, np.ndarray) or not isinstance(yTrain, np.ndarray):
            raise ValueError("Please make sure the type of train_data is np.ndarry")

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

        self._checkSingleOutput(yTrain)

        if xTrain.shape[0] == yTrain.shape[0]:
            self._rawInputCount = xTrain.shape[1]
            xTrain = self.xScaler.fit_transform(xTrain) if self.xScaler else np.copy(xTrain)

            yTrain = self.yScaler.fit_transform(yTrain) if self.yScaler else np.copy(yTrain)

            xTrain = self.polyFeature.transform(xTrain) if self.polyFeature else np.copy(xTrain)

            return xTrain, yTrain

        else:
            raise ValueError("The shapes of x and y are not consistent. Please check them!")

    def _transformX(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("Prediction X must be a 1D sample or a 2D sample matrix.")
        if self._rawInputCount is not None and X.shape[1] != self._rawInputCount:
            raise ValueError(
                f"Prediction X must have {self._rawInputCount} input columns. "
                "A 1D X means one sample; for multiple one-input samples use X.reshape(-1, 1)."
            )

        X = self.xScaler.transform(X) if self.xScaler else X

        X = self.polyFeature.transform(X) if self.polyFeature else X

        return X

    def _transformY(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        Y = self.yScaler.transform(Y) if self.yScaler else Y

        return Y

    def _inverseTransformY(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        Y = self.yScaler.inverse_transform(Y) if self.yScaler else Y

        return Y

    def _inverseTransformX(self, X: np.ndarray) -> np.ndarray:

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

    def applyParameterValues(self, paraList, values, ignoreInactive: bool = True, *, paraInfos=None):
        """
        Apply a flat candidate in encoded parameter coordinates.
        paraInfos fixes its slices across structural changes; by default,
        use the current Setting layout. Apply structure before numeric values.
        """
        if len(paraList) != len(set(paraList)):
            raise ValueError("Parameter names must be unique.")
        if not paraList:
            return self
        if paraInfos is None:
            paraInfos, _, _ = self.setting.getParaInfos(paraList)
        values = np.asarray(values, dtype=object).ravel()
        indices = np.concatenate([paraInfos[name] for name in paraList])
        if indices.size != values.size or not np.array_equal(np.sort(indices), np.arange(values.size)):
            raise ValueError("Candidate size must match the parameter slice dimensions.")

        structuralValues = {}
        for name in paraList:
            if name in self._parameterAppliers:
                value = values[paraInfos[name]]
                if self.setting.isChoicePara(name):
                    value = self.setting.decodeValue(name, value)
                elif value.size == 1:
                    value = value.item()
                structuralValues[name] = value
        for name, value in structuralValues.items():
            self._parameterAppliers[name](value)

        activeInfos = {}
        for name in paraList:
            if name in structuralValues:
                continue
            if name not in self.setting.parVal:
                if ignoreInactive:
                    continue
                raise KeyError(f"Parameter '{name}' is not active for {self.__class__.__name__}.")
            if len(paraInfos[name]) != self.setting.parVal[name].size:
                raise ValueError(f"Candidate width for '{name}' does not match the active parameter dimension.")
            activeInfos[name] = paraInfos[name]
        self.setting.setVals(activeInfos, values)
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
        """
        Prepare raw training data into the canonical prepared-data form.
        """
        xTrain, yTrain = self._checkAndScale(xTrain, yTrain)
        self._prepare_training_components(xTrain)
        return xTrain, yTrain

    def storeTrainingData(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self._checkSingleOutput(yTrain)
        self.xTrain = xTrain
        self.yTrain = yTrain
        return self

    @staticmethod
    def _checkSingleOutput(yTrain):
        values = np.asarray(yTrain)
        if values.ndim not in {1, 2} or (values.ndim == 2 and values.shape[1] != 1):
            raise ValueError(
                "yTrain must have shape (n_samples,) or (n_samples, 1): each surrogate supports a single output. "
                "Use MultiSurrogate for multiple outputs."
            )

    def resetFitState(self):
        self.fitState = {}
        return self

    def requireFitted(self, *stateKeys):
        if self.xTrain is None or self.yTrain is None:
            raise RuntimeError(f"{self.__class__.__name__} has not been fitted yet.")

        missing = [key for key in stateKeys if key not in self.fitState]
        if missing:
            raise RuntimeError(f"{self.__class__.__name__} is missing fitted state: {', '.join(missing)}.")

        return self

    @abc.abstractmethod
    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        Fit the model under the current parameter setting using prepared data.
        """
        pass

    def fitHyper(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        Run model-internal hyper-parameter optimization on prepared data.
        The default implementation directly fits the model under current parameters.
        """
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

    def _format_uncertainty_output(
        self, mean: np.ndarray, var: np.ndarray, returnStd: bool = False, returnVar: bool = False
    ):
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

        # Models supply variance in prepared single-output training units.
        try:
            var = np.broadcast_to(var, mean.shape).copy()
        except ValueError as exc:
            raise ValueError("Prediction variance must match the mean output shape.") from exc
        var = np.maximum(var, 0.0)
        if returnStd:
            std = np.sqrt(var)
            if self.yScaler is not None:
                inverseStd = getattr(self.yScaler, "inverse_transform_std", None)
                if not callable(inverseStd):
                    raise NotImplementedError("yScaler must implement inverse_transform_std for uncertainty output.")
                std = inverseStd(std)
            return mean, std
        if self.yScaler is not None:
            inverseVar = getattr(self.yScaler, "inverse_transform_var", None)
            if not callable(inverseVar):
                raise NotImplementedError("yScaler must implement inverse_transform_var for uncertainty output.")
            var = inverseVar(var)
        return mean, var

    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        self.resetFitState()
        self.xTrain = self.yTrain = None
        self._rawInputCount = None
        try:
            xTrain, yTrain = self.prepareTrainingData(xTrain, yTrain)
            self.fitHyper(xTrain, yTrain)
        except BaseException:
            self.resetFitState()
            self.xTrain = self.yTrain = None
            self._rawInputCount = None
            raise
        return self

    @abc.abstractmethod
    def predict(self, xPred: np.ndarray, returnStd: bool = False, returnVar: bool = False):
        pass


class MultiSurrogate:
    """Train one independent single-output model per target column."""

    def __init__(self, n_surrogates, models_list=None):
        # Own the container while retaining the supplied model instances.
        models_list = [] if models_list is None else list(models_list)
        self.n_surrogates = n_surrogates
        self.rng = np.random.default_rng()

        self.models_list = models_list
        self._validateModels(self.models_list)
        if self.models_list and len(self.models_list) != self.n_surrogates:
            raise ValueError("The number of surrogate models must match n_surrogates.")

    @staticmethod
    def _validateModels(models):
        seen = set()
        for model in models:
            if not isinstance(model, SurrogateABC):
                raise ValueError("Please append the type of surrogate!")
            if id(model) in seen:
                raise ValueError("Each output requires a distinct surrogate instance.")
            seen.add(id(model))

    def append(self, model):
        self._validateModels([*self.models_list, model])
        self.models_list.append(model)

    def _checkCompleteModels(self):
        self._validateModels(self.models_list)
        if len(self.models_list) != self.n_surrogates:
            raise ValueError("The number of models in models_list must match n_surrogates.")

    @property
    def supportsUncertainty(self):
        return (
            bool(self.models_list)
            and len(self.models_list) == self.n_surrogates
            and all(model.supportsUncertainty for model in self.models_list)
        )

    def fit(self, trainX: np.ndarray, trainY: np.ndarray):
        self._checkCompleteModels()
        try:
            trainX = np.asarray(trainX)
            trainY = np.asarray(trainY)
            if trainX.ndim == 1:
                trainX = trainX.reshape(-1, 1)
            if trainY.ndim == 1:
                trainY = trainY.reshape(-1, 1)

            if trainX.ndim != 2 or trainY.ndim != 2:
                raise ValueError("Training inputs and outputs must have a 1D or 2D array shape.")
            if trainY.shape[1] != self.n_surrogates or trainY.shape[1] == 0:
                raise ValueError("The number of outputs in trainY must match n_surrogates.")
            if len(trainX) != len(trainY):
                raise ValueError("Training inputs and outputs must have matching sample counts.")

            for i, model in enumerate(self.models_list):
                model.rng = np.random.default_rng(spawn_seed(self.rng))
                model.fit(trainX, trainY[:, i : i + 1])
        except BaseException:
            for model in self.models_list:
                model.resetFitState()
                model.xTrain = model.yTrain = None
            raise
        return self

    @staticmethod
    def _predictionColumn(values, sampleCount):
        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape != (sampleCount, 1):
            raise ValueError("Each surrogate prediction must have shape (n_predictions, 1).")
        return values

    def predict(self, testX: np.ndarray, returnStd: bool = False, returnVar: bool = False):
        self._checkCompleteModels()
        if returnStd and returnVar:
            raise ValueError("Only one of returnStd and returnVar can be True.")
        if returnStd or returnVar:
            for model in self.models_list:
                model._normalize_predict_flags(returnStd, returnVar)

        sampleCount = len(np.atleast_2d(testX))
        predictions, uncertainties = [], []
        for model in self.models_list:
            if returnStd or returnVar:
                prediction, uncertainty = model.predict(testX, returnStd=returnStd, returnVar=returnVar)
                uncertainties.append(self._predictionColumn(uncertainty, sampleCount))
            else:
                prediction = model.predict(testX)
            predictions.append(self._predictionColumn(prediction, sampleCount))

        mean = np.hstack(predictions)
        if returnStd or returnVar:
            return mean, np.hstack(uncertainties)
        return mean

    def predict_deriv(self, testX: np.ndarray, variables=None, missing=None):
        """Combine original-unit derivatives from models that support them."""
        self._checkCompleteModels()
        methods = [getattr(model, "predict_deriv", None) for model in self.models_list]
        if not all(callable(method) for method in methods):
            raise NotImplementedError("Each surrogate must support predict_deriv to combine derivatives.")
        sampleCount = len(np.atleast_2d(testX))
        derivatives = []
        for method in methods:
            values = method(testX, variables) if missing is None else method(testX, variables, missing=missing)
            values = np.asarray(values)
            if values.ndim != 3 or values.shape[0] != sampleCount or values.shape[2] != 1:
                raise ValueError("Each derivative must have shape (n_predictions, n_variables, 1).")
            if derivatives and values.shape[:2] != derivatives[0].shape[:2]:
                raise ValueError("Derivative sample and variable axes must match across surrogates.")
            derivatives.append(values)
        return np.concatenate(derivatives, axis=2)
