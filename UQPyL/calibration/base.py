import abc
import time
import os

import numpy as np

from . import util as metric_util
from ..core.params import Params
from ..problem import ModelProblem
from .runtime import CalState, format_summary, save_log


class CalibrationABC(metaclass=abc.ABCMeta):
    """
    Abstract base class for calibration methods.

    Calibration methods consume `ModelProblem`, where observations and masks
    are part of the problem definition and simulations are aligned with the
    same observation space.

    Examples:
        >>> from UQPyL.calibration import GLUE
        >>> method = GLUE(metric="rmse", verboseFlag=False)
        >>> result = method.run(problem, X, threshold=0.5)
        >>> print(result.bestDecs)

    Notes:
        - Calibration methods only accept `ModelProblem`.
        - `problem.simFunc(X)` must return a simulation tensor with shape
          `(n_samples, n_time, n_series)`.
        - Internally, simulations and observations are flattened into the
          shared observation space `(n_samples, n_obs)` for scoring, masking,
          and update formulas.
    """

    def __init__(
        self,
        verboseFlag: bool = False,
        verboseFreq: int = 1,
        saveFlag: bool = False,
        logFlag: bool = False,
        metric="rmse",
    ):
        self.verboseFlag = verboseFlag
        self.verboseFreq = verboseFreq
        self.saveFlag = saveFlag
        self.logFlag = logFlag
        self.params = Params()
        self.setting = self.params
        self.result = CalState(self)
        self.state = self.result
        self.metric, self.metricHigherIsBetter = self._resolve_metric(metric)
        self.metricName = metric if isinstance(metric, str) else getattr(metric, "__name__", "custom_metric")
        self.set("verboseFlag", verboseFlag)
        self.set("verboseFreq", verboseFreq)
        self.set("saveFlag", saveFlag)
        self.set("logFlag", logFlag)
        self.set("metric", self.metricName)

    def set(self, key, value):
        self.params.set(key, value)

    def get(self, *args):
        return self.params.get(*args)

    def setProblem(self, problem: ModelProblem):
        if not isinstance(problem, ModelProblem):
            raise TypeError("Calibration methods only accept ModelProblem.")
        self.problem = problem

    def setup(self, problem: ModelProblem):
        self.setProblem(problem)
        self.result.reset()
        self.state = self.result
        self.workDir = getattr(problem, "workDir", os.getcwd())

    def finalize(self):
        result = self.state.buildResult()
        summaryText = format_summary(result)
        if self.verboseFlag:
            print(summaryText)
        if self.logFlag:
            save_log(result, self.workDir)
        return result

    def run(self, problem: ModelProblem, *args, **kwargs):
        """
        Execute the calibration workflow.

        Args:
            problem: Model problem containing parameter space, observations,
                masks, and `simFunc`.
            *args: Method-specific positional arguments.
            **kwargs: Method-specific keyword arguments.

        Returns:
            CalResult: Final calibration result.
        """
        self.setup(problem)
        start = time.perf_counter()
        self._runCore(problem, *args, **kwargs)
        self.state.runtime = time.perf_counter() - start
        return self.finalize()

    def getFlattenedObs(self):
        return self.problem.flattenObs()

    def getFlattenedMask(self):
        return self.problem.flattenMask()

    def getValidMask(self):
        return ~self.getFlattenedMask()

    def getValidObs(self):
        obs = self.getFlattenedObs()
        return obs[self.getValidMask()]

    def evaluate(self, X, validOnly: bool = True):
        sim = self.problem.simFunc(X)
        sim2d = self.problem.flattenSim(sim)
        if not validOnly:
            return sim2d
        return sim2d[:, self.getValidMask()]

    def recordBest(self, decs: np.ndarray, sim: np.ndarray):
        self.state.bestDecs = np.asarray(decs).copy()
        self.state.bestSim = np.asarray(sim).copy()

    def recordPosterior(self, decs: np.ndarray, sim: np.ndarray):
        self.state.posteriorDecs = np.asarray(decs).copy()
        self.state.posteriorSims = np.asarray(sim).copy()

    def recordBehavioral(self, decs: np.ndarray, sim: np.ndarray):
        self.state.behavioralDecs = np.asarray(decs).copy()
        self.state.behavioralSims = np.asarray(sim).copy()

    def recordElite(self, decs: np.ndarray, sim: np.ndarray):
        self.state.eliteDecs = np.asarray(decs).copy()
        self.state.eliteSims = np.asarray(sim).copy()

    def score(self, sim: np.ndarray):
        """
        Evaluate the configured metric on flattened simulations.

        Args:
            sim: Simulation matrix with shape `(n_samples, n_obs)`.

        Returns:
            np.ndarray: Metric values for each simulation row.
        """
        obs = self.getFlattenedObs()
        mask = self.getFlattenedMask()
        return np.asarray(self.metric(obs, sim, mask=mask), dtype=float)

    def normalizedScore(self, sim: np.ndarray):
        raw = self.score(sim)
        if self.metricHigherIsBetter:
            return -raw
        return raw

    def _resolve_metric(self, metric):
        if callable(metric):
            return metric, False

        metricMap = {
            "mse": (metric_util.mse, False),
            "mae": (metric_util.mae, False),
            "rmse": (metric_util.rmse, False),
            "nse": (metric_util.nse, True),
            "r2": (metric_util.r2, True),
            "pbias": (metric_util.pbias, False),
            "pearson_r": (metric_util.pearson_r, True),
            "kge": (metric_util.kge, True),
        }
        if metric not in metricMap:
            raise ValueError(f"Unsupported metric '{metric}'.")
        return metricMap[metric]

    @abc.abstractmethod
    def _runCore(self, problem: ModelProblem, *args, **kwargs):
        pass
