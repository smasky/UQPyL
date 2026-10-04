import numpy as np
from itertools import product
from functools import wraps
from copy import deepcopy
from inspect import signature, Parameter
from time import perf_counter

from .base import SurrogateABC
from ..optimization.base import AlgorithmABC
from .split import RandSelect, _resolveRng
from .metric import r_square
from ..problem.problem import Problem
from ..core import spawn_seed


def _invalidateOnFailure(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._startReport(method.__name__)
        start = perf_counter()
        try:
            result = method(self, *args, **kwargs)
            self.lastReport["status"] = "finished"
            return result
        except BaseException as error:
            self.lastReport.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error_type=type(error).__name__,
                message=str(error),
            )
            # Preprocessing and parameters may already have changed. As with
            # public fit(), failed tuning requires a fresh fit before prediction.
            self.model.resetFitState()
            self.model.xTrain = self.model.yTrain = None
            raise
        finally:
            self.lastReport["elapsed_seconds"] = perf_counter() - start

    return wrapped


class AutoTuner:
    """
    Hyper-parameter tuner for surrogate models.

    The tuner evaluates candidate parameter settings by fitting the target
    surrogate on a train split and scoring predictions on a validation split.
    It supports both optimizer-driven tuning (`optTune`) and explicit grid
    search (`gridTune`).
    Failed or interrupted tuning invalidates the model; refit before prediction.

    Examples:
        >>> tuner = AutoTuner(model, optimizer)
        >>> bestParams, bestScore = tuner.optTune(xData, yData)
    """

    def __init__(self, model: SurrogateABC, optimizer: AlgorithmABC = None):
        """
        Initialize the AutoTuner

        Args:
            model: Surrogate, the surrogate model
            optimizer: Algorithm, the optimizer
        """
        self.optimizer = optimizer

        self.model = model
        self.rng = np.random.default_rng()
        self.lastSplit = None
        self.candidateFailures = []
        self._candidateCount = 0
        self._modelSeed = None
        self.lastReport = None

    def _startReport(self, method):
        self.lastSplit = None
        self.candidateFailures = []
        self._candidateCount = 0
        self._lastFit = None
        self.lastReport = {
            "method": method,
            "status": "running",
            "candidates": [],
            "fit_calls": 0,
            "tracked_objective_evaluations": 0,
            "best_candidate_index": None,
            "best_validation_score": None,
        }

    def getReport(self):
        """Return an independent diagnostic report for the latest tuning call."""
        return deepcopy(self.lastReport)

    @staticmethod
    def _reportValue(value):
        if isinstance(value, np.ndarray):
            return AutoTuner._reportValue(value.tolist())
        if isinstance(value, np.generic):
            return AutoTuner._reportValue(value.item())
        if isinstance(value, (list, tuple)):
            return [AutoTuner._reportValue(item) for item in value]
        if isinstance(value, dict):
            return {key: AutoTuner._reportValue(item) for key, item in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {
            "type": type(value).__name__,
            "name": getattr(value, "displayName", getattr(value, "__name__", type(value).__name__)),
        }

    def _parameterSnapshot(self):
        values = {
            name: self.model.setting.get(name) for name in (*self.model.setting.parVal, *self.model.setting.parCon)
        }
        values.update({name: getattr(self.model, name, None) for name in self.model._parameterAppliers})
        return self._reportValue(values)

    def _splitData(self, xRaw, ratio, seed, rng, *, splitter=None, splitIndices=None):
        self.candidateFailures = []
        self._candidateCount = 0
        parent = self.rng if seed is None and rng is None else _resolveRng(seed, rng)
        splitSeed, modelSeed, optimizerSeed = (spawn_seed(parent) for _ in range(3))
        if splitter is not None and splitIndices is not None:
            raise ValueError("Provide only one of splitter or splitIndices.")
        strategy = "random_holdout"
        if splitIndices is not None:
            indices = splitIndices
            strategy = "fixed_indices"
        elif splitter is not None:
            split = getattr(splitter, "split", splitter)
            if not callable(split):
                raise TypeError("splitter must be callable or provide a split method.")
            parameters = signature(split).parameters
            if "seed" in parameters or any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values()):
                indices = split(xRaw.copy(), seed=splitSeed)
            elif "rng" in parameters:
                indices = split(xRaw.copy(), rng=np.random.default_rng(splitSeed))
            else:
                indices = split(xRaw.copy())
            strategy = "custom_splitter"
        else:
            indices = RandSelect(ratio).split(xRaw, seed=splitSeed)
        trainIdx, testIdx = self._validateSplit(indices, len(xRaw))
        self._modelSeed = modelSeed
        self.lastSplit = {
            "seed": None if seed is None else int(seed),
            "split_seed": splitSeed,
            "model_seed": modelSeed,
            "optimizer_seed": optimizerSeed,
            "train_indices": trainIdx.copy(),
            "test_indices": testIdx.copy(),
            "strategy": strategy,
        }
        self.lastReport["split"] = self._reportValue(self.lastSplit)
        return trainIdx, testIdx

    @staticmethod
    def _validateSplit(indices, nSamples):
        if not isinstance(indices, (list, tuple)) or len(indices) != 2:
            raise ValueError("Provide a single (train_indices, validation_indices) pair; select a fold explicitly.")
        checked = []
        for values in indices:
            try:
                values = np.asarray(values)
            except ValueError as error:
                raise ValueError(
                    "Split indices must be one-dimensional integer arrays; select a single fold."
                ) from error
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise ValueError("Split indices must be one-dimensional integer arrays; select a single fold.")
            if not len(values) or np.any(values < 0) or np.any(values >= nSamples):
                raise ValueError("Split indices must be nonempty and within the data range.")
            if len(np.unique(values)) != len(values):
                raise ValueError("Split indices must not contain duplicates.")
            checked.append(values.astype(np.intp, copy=True))
        if np.intersect1d(*checked).size:
            raise ValueError("Training and validation indices must be disjoint.")
        return tuple(checked)

    def _prepareTuningData(self, xData, yData, ratio, seed, rng, splitter, splitIndices, tuneMode):
        if tuneMode not in ("joint", "separate"):
            raise ValueError("tuneMode must be either 'joint' or 'separate'.")
        self.lastReport["tune_mode"] = tuneMode
        xRaw, yRaw = np.asarray(xData), np.asarray(yData)
        if xRaw.ndim == 1:
            xRaw = xRaw.reshape(-1, 1)
        if yRaw.ndim == 1:
            yRaw = yRaw.reshape(-1, 1)
        if xRaw.ndim != 2 or yRaw.ndim != 2 or len(xRaw) != len(yRaw):
            raise ValueError("Tuning inputs and outputs must be 1D/2D arrays with matching sample counts.")
        self.model._checkSingleOutput(yRaw)
        trainIdx, testIdx = self._splitData(xRaw, ratio, seed, rng, splitter=splitter, splitIndices=splitIndices)
        self._validateValidation(yRaw[testIdx])
        xTrain, yTrain = self.model.prepareTrainingData(xRaw[trainIdx], yRaw[trainIdx])
        self.model.storeTrainingData(xTrain, yTrain)
        self._initialize_model_components(xTrain)
        self.lastReport.update(training_samples=len(trainIdx), validation_samples=len(testIdx), refit_samples=len(xRaw))
        return xRaw, yRaw, xTrain, yTrain, xRaw[testIdx], yRaw[testIdx]

    def _validateValidation(self, values):
        if len(values) < 2:
            raise ValueError("R2 validation requires at least two samples; increase ratio or supply more data.")
        if not np.all(np.isfinite(values)):
            raise ValueError("R2 validation outputs must be finite.")
        if np.all(values == values[0]):
            raise ValueError("R2 validation requires nonconstant outputs with finite variation.")

    def _noValidCandidate(self):
        self.model.resetFitState()
        self.model.xTrain = self.model.yTrain = None
        raise RuntimeError("No candidate produced a finite validation score.")

    def _initialize_model_components(self, xData: np.ndarray):
        kernel = getattr(self.model, "kernel", None)
        if kernel is not None and hasattr(kernel, "initialize"):
            kernel.initialize(xData.shape[1])

    def _fit_with_mode(self, xTrain, yTrain, tuneMode):
        # Count existing objective calls; reporting never evaluates the model itself.
        if self._modelSeed is not None:
            self.model.rng = np.random.default_rng(self._modelSeed)
        original = getattr(self.model, "_objfunc", None)
        hadOwnObjective = "_objfunc" in self.model.__dict__
        calls = 0

        def objective(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        trace = {
            "entry_point": "fitModel" if tuneMode == "joint" else "fitHyper",
            "status": "running",
            "parameters_before_fit": self._parameterSnapshot(),
        }
        self._lastFit = trace
        if callable(original):
            self.model._objfunc = objective
        start = perf_counter()
        self.lastReport["fit_calls"] += 1
        try:
            if tuneMode == "joint":
                self.model.fitModel(xTrain, yTrain)
            else:
                self.model.fitHyper(xTrain, yTrain)
            trace["status"] = "finished"
        except BaseException as error:
            trace.update(status="failed", error_type=type(error).__name__, message=str(error))
            raise
        finally:
            solver = self.model.fitState.get("solver")
            if solver is not None:
                trace["solver"] = {
                    "iterations": solver["iterations"],
                    "max_iterations": solver["maxIterations"],
                    "iteration_limit_reached": solver["iterationLimitReached"],
                }
            trace.update(
                elapsed_seconds=perf_counter() - start,
                objective_evaluations=calls if callable(original) else None,
                parameters_after_fit=self._parameterSnapshot(),
            )
            self.lastReport["tracked_objective_evaluations"] += calls
            if callable(original):
                if hadOwnObjective:
                    self.model._objfunc = original
                else:
                    del self.model._objfunc
        return trace

    def _scoreCandidate(self, xTrain, yTrain, xTest, yTest, tuneMode, *, candidate):
        candidateIndex = self._candidateCount
        self._candidateCount += 1
        record = {
            "candidate_index": candidateIndex,
            "candidate_encoded": self._reportValue(candidate),
            "status": "running",
            "validation_score": None,
        }
        self.lastReport["candidates"].append(record)
        start = perf_counter()
        try:
            self._fit_with_mode(xTrain, yTrain, tuneMode)
            prediction = self.model.predict(xTest)
            if not np.all(np.isfinite(prediction)):
                raise FloatingPointError("Candidate predictions are not finite.")
            score = float(r_square(yTest, prediction))
            if not np.isfinite(score):
                raise FloatingPointError("Candidate validation score is not finite.")
            record.update(status="finished", validation_score=score)
            best = self.lastReport["best_validation_score"]
            if best is None or score > best:
                self.lastReport.update(best_validation_score=score, best_candidate_index=candidateIndex)
            return score
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__, message=str(error))
            if not isinstance(error, (np.linalg.LinAlgError, ArithmeticError)):
                raise
            self.candidateFailures.append(
                {"candidate_index": candidateIndex, "error_type": type(error).__name__, "message": str(error)}
            )
            return -np.inf
        finally:
            record["fit"] = deepcopy(self._lastFit)
            record["elapsed_seconds"] = perf_counter() - start

    def _refitSelected(self, xRaw, yRaw, tuneMode, selected):
        self.lastReport["selected_candidate_encoded"] = self._reportValue(selected)
        refit = {"status": "preparing"}
        self.lastReport["final_refit"] = refit
        fitStarted = False
        try:
            xFull, yFull = self.model.prepareTrainingData(xRaw, yRaw)
            fitStarted = True
            self._fit_with_mode(xFull, yFull, tuneMode)
        except BaseException as error:
            if fitStarted:
                refit.update(deepcopy(self._lastFit))
            refit.update(status="failed", error_type=type(error).__name__, message=str(error))
            raise
        else:
            self.lastReport["final_refit"] = deepcopy(self._lastFit)
            self.lastReport["final_parameters"] = self._parameterSnapshot()

    def _resolve_para_list(self, paraList=None, owner=None):
        if paraList is not None:
            return list(paraList)

        paraList = self.model.setting.getParaList(owner=owner, tunableOnly=True)
        if not paraList:
            ownerMsg = "" if owner is None else f" for owner '{owner}'"
            raise ValueError(f"No tunable parameters found{ownerMsg}.")

        return paraList

    @_invalidateOnFailure
    def optTune(
        self,
        xData: np.ndarray,
        yData: np.ndarray,
        paraList: list = None,
        ratio: int = 10,
        owner: str = None,
        tuneMode: str = "separate",
        *,
        seed=None,
        rng=None,
        splitter=None,
        splitIndices=None,
    ):
        """
        Optimize the hyper-parameters for the surrogate model

        Args:
            xData: Raw input matrix with shape (n_samples, n_input).
            yData: Raw output matrix with shape (n_samples, n_output).
            paraList: list, optional parameter names to tune
            ratio: Validation percentage used only by the default random split.
            owner: str, optional owner filter such as `model` or `kernel`
            tuneMode: 'joint' fits exact candidates; 'separate' allows internal tuning.
            seed: Optional local random seed; mutually exclusive with rng.
            rng: Optional NumPy Generator.
            splitter: Callable or object with split(X), returning one index pair.
            splitIndices: Fixed (train_indices, validation_indices); excludes splitter.

        Returns:
            tuple: Final refitted parameter vector and best held-out R-squared.

        Notes:
            Preprocessing is fitted on training rows for selection, then all rows
            for the final refit. The supplied model is mutated and invalidated on
            failure. getReport() exposes candidate and final-fit differences.
        """
        if not callable(getattr(self.optimizer, "run", None)):
            raise TypeError("optTune requires an optimizer with a run method.")
        if isinstance(self.optimizer, AlgorithmABC):
            self.optimizer._checkObjectiveCount(1)
        xRaw, yRaw, xTrain, yTrain, xTestRaw, yTestRaw = self._prepareTuningData(
            xData, yData, ratio, seed, rng, splitter, splitIndices, tuneMode
        )
        paraList = self._resolve_para_list(paraList=paraList, owner=owner)

        paraInfos, ub, lb = self.model.setting.getParaInfos(paraList)
        nInput = ub.size
        encodings = {name: (self.model.setting.parType[name], self.model.setting.parLog[name]) for name in paraList}

        def applyCandidate(values):
            self.model.applyParameterValues(paraList, values, paraInfos=paraInfos)
            # An optimizer has one fixed box; switching kernels must not
            # silently reinterpret its coordinates or parameter bounds.
            for name, indices in paraInfos.items():
                setting = self.model.setting
                if name not in setting.parVal:
                    continue
                _, currentUpper, currentLower = setting.getParaInfos([name])
                if (
                    (setting.parType[name], setting.parLog[name]) != encodings[name]
                    or not np.array_equal(currentUpper, ub[indices])
                    or not np.array_equal(currentLower, lb[indices])
                ):
                    raise ValueError(
                        f"Parameter '{name}' changed bounds or encoding during optTune; "
                        "use compatible kernel settings or separate searches."
                    )

        hasValidCandidate = False

        def objFunc(X):
            nonlocal hasValidCandidate

            Y = np.zeros((X.shape[0], 1))

            XX = X.copy()

            for i, x in enumerate(XX):
                applyCandidate(x)

                obj = self._scoreCandidate(
                    xTrain,
                    yTrain,
                    xTestRaw,
                    yTestRaw,
                    tuneMode,
                    candidate={name: x[indices] for name, indices in paraInfos.items()},
                )
                hasValidCandidate = hasValidCandidate or np.isfinite(obj)

                Y[i, 0] = obj

            return Y

        problem = Problem(nInput=nInput, nObj=1, ub=ub, lb=lb, objFunc=objFunc, optType="max")

        res = self.optimizer.run(problem=problem, seed=self.lastSplit["optimizer_seed"])
        bestTrueDecs = np.asarray(res.bestDecs).ravel()
        bestTrueObj = np.asarray(res.bestObjs).ravel()
        if not hasValidCandidate or not np.all(np.isfinite(bestTrueObj)):
            self._noValidCandidate()

        applyCandidate(bestTrueDecs)

        self._refitSelected(xRaw, yRaw, tuneMode, {name: bestTrueDecs[indices] for name, indices in paraInfos.items()})

        return self.model.getParameterValues(*paraList, ignoreInactive=True), bestTrueObj

    def _applyGridCandidate(self, paraList, candidate):
        # Grid entries are grouped by parameter, so a vector is one candidate.
        parts = [np.asarray(value, dtype=object).ravel() for value in candidate]
        offsets = np.cumsum([0] + [part.size for part in parts])
        paraInfos = {name: np.arange(offsets[i], offsets[i + 1]) for i, name in enumerate(paraList)}
        self.model.applyParameterValues(paraList, np.concatenate(parts), paraInfos=paraInfos)

    def _validateGridNames(self, paraList, choices):
        knownNames = set(self.model.setting.parVal) | set(self.model._parameterAppliers)
        structuralNames = [name for name in paraList if name in self.model._parameterAppliers]
        if structuralNames and any(name not in knownNames for name in paraList):
            structuralChoices = [choices[paraList.index(name)] for name in structuralNames]
            # Probe structure on private copies: a parameter may become active
            # only after a kernel/loss switch later in this grid.
            for combination in product(*structuralChoices):
                probe = deepcopy(self.model)
                parts = [np.asarray(value, dtype=object).ravel() for value in combination]
                offsets = np.cumsum([0] + [part.size for part in parts])
                infos = {name: np.arange(offsets[i], offsets[i + 1]) for i, name in enumerate(structuralNames)}
                probe.applyParameterValues(structuralNames, np.concatenate(parts), paraInfos=infos)
                knownNames.update(probe.setting.parVal)
        unknownNames = [name for name in paraList if name not in knownNames]
        if unknownNames:
            raise ValueError(f"Unknown or non-tunable grid parameters: {unknownNames}.")

    @_invalidateOnFailure
    def gridTune(
        self,
        xData: np.ndarray,
        yData: np.ndarray,
        paraGrid: dict = None,
        ratio: int = 10,
        owner: str = None,
        tuneMode: str = "separate",
        *,
        seed=None,
        rng=None,
        splitter=None,
        splitIndices=None,
    ):
        """
        Grid search for the best parameter combination

        Args:
            xData: Raw input matrix with shape (n_samples, n_input).
            yData: Raw output matrix with shape (n_samples, n_output).
            paraGrid: dict, optional parameter grid
            ratio: Validation percentage used only by the default random split.
            owner: str, optional owner filter used when paraGrid is not provided
            tuneMode: 'joint' fits exact candidates; 'separate' allows internal tuning.
            seed: Optional local random seed; mutually exclusive with rng.
            rng: Optional NumPy Generator.
            splitter: Callable or object with split(X), returning one index pair.
            splitIndices: Fixed (train_indices, validation_indices); excludes splitter.

        Returns:
            tuple: Final refitted parameter values and best held-out R-squared.

        Notes:
            Candidates use the Setting's encoded parameter coordinates. The model
            is refitted on all rows after selection and invalidated on failure.
            getReport() exposes candidates, actual fitted parameters, and fit cost.
        """
        xRaw, yRaw, xTrain, yTrain, xTestRaw, yTestRaw = self._prepareTuningData(
            xData, yData, ratio, seed, rng, splitter, splitIndices, tuneMode
        )

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
        self._validateGridNames(paraList, choices)
        paraCombs = product(*choices)

        # Grid search
        bestObj = -np.inf
        bestDecs = None

        for paraComb in paraCombs:
            self._applyGridCandidate(paraList, paraComb)

            obj = self._scoreCandidate(
                xTrain, yTrain, xTestRaw, yTestRaw, tuneMode, candidate=dict(zip(paraList, paraComb))
            )

            if obj > bestObj:
                bestObj = obj
                bestDecs = paraComb

        if bestDecs is None:
            self._noValidCandidate()
        self._applyGridCandidate(paraList, bestDecs)

        self._refitSelected(xRaw, yRaw, tuneMode, dict(zip(paraList, bestDecs)))

        return self.model.getParameterValues(*paraList, ignoreInactive=True), bestObj
