import abc
import time
from typing import List, Union

import numpy as np

from ..core.params import Params
from ..core import config
from ..core.runtime_session import RunSession
from ..core.runtime_lifecycle import RunLifecycle
from ..problem import ProblemABC as Problem
from .runtime import AnaState, SqliteStorage, Verbose

AnaIndex = Union[str, int, List[int]]

class AnalysisABC(RunLifecycle, metaclass=abc.ABCMeta):
    """
    Abstract base class for analysis methods.
    Shared workflow and utilities for sensitivity analysis methods.
    """

    _runEntryPoint = "analyze"

    def __init__(self, verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the analysis base class.

        Args:
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """

        # Set flags for verbosity, logging, and saving
        self.verboseFlag = verboseFlag
        self.logFlag = logFlag
        self.saveFlag = saveFlag
        
        # Initialize settings and results
        self.params = Params()
        self.state = AnaState(self)
        self.storage = None
        self.session: RunSession | None = None
        self.runId = None
        
    def set(self, key, value):
        """
        Set an analysis parameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        
        self.params.set(key, value)
    
    def get(self, *args):
        """
        Retrieve one or more analysis parameters.

        Args:
            *args: Parameter names.

        Returns:
            The requested parameter value or values.
        """
        
        return self.params.get(*args)
        
    def setProblem(self, problem: Problem):
        """
        Set the problem instance for the analysis.

        Args:
            problem: Problem instance defining the input and output space.
        """
        
        self.problem = problem

    def setup(self, problem):
        self.setProblem(problem)
        self._startRun()
        self.state.reset()
        self.session = None
        Verbose.setupContext(self, problem)
        if self.saveFlag:
            rootDir = config.resolveWorkDir(getattr(problem, "workDir", None), Verbose.workDir)
            self.storage = SqliteStorage(rootDir)
            self.session = self.storage.create_run(self)
            self.runId = self.session.run_id

    def finalize(self):
        result = self.state.buildResult()
        if self.saveFlag and self.session is not None:
            self.storage.saveResult(self.session, result)
            self._closeStandaloneSession()
        Verbose.printConclusion(self, result)
        if self.logFlag:
            Verbose.saveLog(self)
        return result

    def analyze(self, problem, *args, **kwargs):
        """
        Run the analysis workflow and return the final `AnaResult`.

        Expected public inputs follow the unified protocol:
        `analyze(problem, X, Y=None, meta=None, target="objs", index="all")`.
        Here `target` is the semantic label of `Y`, and when `Y` is not
        provided it also selects which problem output block to evaluate.
        """
        meta = kwargs.get("meta")
        if meta is not None:
            self.checkMeta(meta)
        self.setup(problem)
        Verbose.printSettings(self)
        start = time.perf_counter()
        self._analyzeCore(problem, *args, **kwargs)
        self.state.runtime = time.perf_counter() - start
        return self.finalize()

    def checkMeta(self, meta):
        """
        Validate sampling metadata produced by `sampleWithMeta()`.
        """
        return None

    def check_Y(self, X, Y, target: str = 'objs', index: AnaIndex = 'all'):
        """
        Resolve and slice analysis outputs.

        `target` labels the meaning of `Y`, typically `objs` or `cons`.
        If `Y` is not provided, `target` also selects which problem output
        block should be evaluated. If `index` is not `'all'`, only the
        selected output columns are kept.
        """
        if Y is None:
            Y = self.evaluate(X, target=target)
        X, Y = self.__check_X_Y__(X, Y)
        if target not in ('objs', 'cons'):
            raise ValueError("Target must be 'objs' or 'cons'.")
        labels = getattr(self.problem, 'objLabels' if target == 'objs' else 'conLabels', None)
        prefix = 'obj' if target == 'objs' else 'con'
        labels = list(labels) if labels is not None and len(labels) == Y.shape[1] else [
            f'{prefix}{i + 1}' for i in range(Y.shape[1])]
        if not (isinstance(index, str) and index == 'all'):
            indices = self._normalize_index(index)
            try:
                Y = Y[:, indices]
                labels = [labels[i] for i in indices]
            except (IndexError, TypeError):
                raise ValueError("Please check the index you set!")
        self.outputLabels = labels
        return Y
    
    def recordResult(self, X, Y, res, target: str = 'objs', meta=None):
        self.state.record(X, Y, res, target=target, meta=meta)
        for metric in self.state.metrics:
            for i, target in enumerate(metric.rowLabels):
                self.record(target, metric.name, metric.colLabels, metric.values[i])

    def record(self, target, indicator, labels, values):
        """
        Record the analysis results.

        Args:
            target: Output label such as `obj1` or `con1`.
            indicator: Metric name.
            labels: Input variable labels.
            values: Metric values.
        """
                        
        self.state.verbose.setdefault(target, {})
        self.state.verbose[target].setdefault(indicator, {})
        
        for label, v in zip(labels, values):
            self.state.verbose[target][indicator][label] = v

        self.state.verbose[target][indicator]['array'] = np.array(values)
        

    def __check_X_Y__(self, X, Y):
        """
        Check input and output arrays.

        Args:
            X: Input matrix.
            Y: Output matrix.

        Returns:
            The validated `X` and `Y`.
        """
        
        if not isinstance(X, np.ndarray) and X is not None:
            raise TypeError("X must be an instance of np.ndarray or None!")
        
        if not isinstance(Y, np.ndarray) and Y is not None:
            raise TypeError("Y must be an instance of np.ndarray or None!")

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if X is None or X.ndim != 2 or X.shape[1] != self.problem.nInput:
            raise ValueError("X must have shape (nSamples, nInput).")
        if Y.ndim != 2 or Y.shape[0] != X.shape[0] or Y.shape[1] == 0:
            raise ValueError("Y must have one row per X sample and at least one output column.")
                  
        return X, Y
    
    def evaluate(self, X, target: str = 'objs'):
        """
        Evaluate the problem with the given input data.

        Args:
            X: Input matrix.
            target: Semantic output label to evaluate, typically `objs` or `cons`.

        Returns:
            The requested output matrix.
        """

        if target not in ('objs', 'cons'):
            raise ValueError("Target must be 'objs' or 'cons'!")

        evalRes = self.problem.evaluate(X, target=target)
        Y = evalRes.objs if target == 'objs' else evalRes.cons
        if Y is None:
            raise ValueError(f"Problem does not provide target '{target}'.")
        return Y

    def _normalize_index(self, index: AnaIndex):
        """
        Normalize output column selection into a list of integers.
        """
        if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)):
            return [index]
        if isinstance(index, (list, tuple, np.ndarray)):
            indices = list(index)
            if indices and all(isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_)) for i in indices):
                return indices
        raise ValueError("Index must be 'all', an integer, or a list of integers!")
    
    @abc.abstractmethod
    def _analyzeCore(self, problem, *args, **kwargs):
        pass
