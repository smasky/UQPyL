import abc
import os
import time

import numpy as np

from .chain import Chain
from .runtime import InfResult, Result, SqliteStorage, Verbose
from ..doe import LHS
from ..problem import ProblemABC


class InferenceABC(metaclass=abc.ABCMeta):
    """
    Abstract base class for inference methods.
    Shared workflow and utilities for MCMC-style sampling methods.
    """

    def __init__(
        self,
        maxIters: int = 1000,
        verboseFlag: bool = True,
        verboseFreq: int = 10,
        logFlag: bool = False,
        saveFlag: bool = False,
        saveFreq: int = 100,
        logProbFunc=None,
        maxInitAttempts: int = 1000,
    ):
        """
        Initialize the inference base class with runtime flags and optional hooks.

        Args:
            maxIters: Number of formal sampling draws, including the initial draw.
            verboseFlag: Whether to print compact runtime summaries.
            verboseFreq: Iteration interval for terminal and log summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist snapshots and final result to sqlite.
            saveFreq: Iteration interval for sqlite snapshots.
            logProbFunc: Optional custom log-probability function.
            maxInitAttempts: Maximum LHS batches used to find feasible initial chains.
        """
        # Initialize settings and results
        self.params = Params()
        self.setting = self.params
        self.result = Result(self)
        self.state = self.result

        # Set runtime flags and hooks
        self.problem = None
        self.maxIters = maxIters
        self.verboseFlag = verboseFlag
        self.verboseFreq = verboseFreq
        self.logFlag = logFlag
        self.saveFlag = saveFlag
        self.saveFreq = saveFreq
        self.logProbFunc = logProbFunc
        self.maxInitAttempts = maxInitAttempts
        self.storage = None
        self.storageCtx = None
        self.runId = None
        self.setParaVal("maxInitAttempts", maxInitAttempts)
        if logProbFunc is not None:
            self.setParaVal("logProbFunc", getattr(logProbFunc, "__name__", repr(logProbFunc)))

    def run(self):
        """
        Run the inference workflow.
        """
        pass

    def setup(self, problem: ProblemABC, seed: int = None):
        """
        Set the problem, reset runtime state, and initialize optional storage.

        Args:
            problem: Problem instance defining the inference space.
            seed: Optional random seed.
        """
        self.setProblem(problem)
        self.reset()
        self.validateProblem()
        Verbose.setupContext(self, problem)

        if self.saveFlag:
            rootDir = getattr(problem, "workDir", os.getcwd())
            self.storage = SqliteStorage(rootDir)

        # Initialize random seed
        if seed is None:
            seed = np.random.randint(0, 1000000)
        np.random.seed(seed)

        self.setParaVal("seed", seed)
        self.setParaVal("saveFreq", self.saveFreq)

        if self.saveFlag:
            self.storageCtx = self.storage.createRun(self)
            self.runId = self.storageCtx["runId"]

        Verbose.printSettings(self)

    def validateProblem(self):
        """
        Validate problem-level assumptions shared by inference algorithms.
        """
        if self.problem.nOutput != 1:
            raise ValueError("Inference currently supports scalar objectives only.")

    def reset(self):
        """
        Reset counters and runtime state before one run.
        """
        self.FEs = 0
        self.iters = 0
        self.iter = 0
        self.startTime = time.perf_counter()
        self.state.reset()

    def initialSampling(self, problem: ProblemABC, nChains: int, seed: int = None):
        """
        Generate initial samples for all chains.

        For constrained problems, only feasible samples are kept.
        """
        sampler = LHS()
        if problem.nCons == 0:
            sampleSeed = np.random.randint(0, 1000000) if seed is None else seed
            X0 = sampler.sample(self.problem, nChains, sampleSeed)
            objs0, cons0 = self.evaluate(X0)
            return X0, objs0, cons0

        xs = []
        objs = []
        cons = []
        attempts = 0
        maxAttempts = self.getParaVal("maxInitAttempts")
        while len(xs) < nChains and attempts < maxAttempts:
            sampleSeed = np.random.randint(0, 1000000)
            XBatch = sampler.sample(self.problem, nChains, sampleSeed)
            objsBatch, consBatch = self.evaluate(XBatch)
            if consBatch is None:
                raise ValueError("Constrained inference problem must return constraint values.")
            feasible = (consBatch <= 0).all(axis=1)
            for x, obj, con in zip(XBatch[feasible], objsBatch[feasible], consBatch[feasible]):
                xs.append(x)
                objs.append(obj)
                cons.append(con)
                if len(xs) == nChains:
                    break
            attempts += 1

        if len(xs) < nChains:
            raise ValueError(
                f"Unable to initialize {nChains} feasible chains after {maxAttempts} LHS batches."
            )

        return np.asarray(xs), np.asarray(objs), np.asarray(cons)

    def initChains(self, nChains: int, X: np.ndarray, objs: np.ndarray, cons: np.ndarray = None):
        """
        Initialize chain containers from current states.
        """
        nInput = self.problem.nInput
        nOutput = self.problem.nOutput
        nCons = self.problem.nCons
        chains = [Chain(nInput, nOutput, nCons, self.maxIters) for _ in range(nChains)]
        logProb = self.log_prob(objs, decs=X, cons=cons)

        for i, chain in enumerate(chains):
            chain.add(
                X[i],
                objs[i],
                cons[i] if nCons > 0 else None,
                logProb=logProb[i],
                accepted=True,
            )

        return chains

    def update(self, chains):
        """
        Update runtime state from chains and handle progress output.
        """
        self.state.runtime = time.perf_counter() - self.startTime
        self.state.update(chains, self.problem, self.FEs, self.iters)
        if self.verboseFlag or self.logFlag:
            Verbose.printIteration(self)
        if self.saveFlag and self.storageCtx is not None and self.iters % self.saveFreq == 0:
            self.storage.saveSnapshot(self.storageCtx, self, self.buildResult(), isFinal=False)
        return self.state

    def checkTermination(self, chains=None):
        """
        Advance the sampling iteration counter.
        """
        if self.iters >= self.maxIters - 1:
            return False
        self.iters += 1
        self.iter = self.iters
        return True

    def buildResult(self):
        """
        Build the final `InfResult`.
        """
        result = self.state.buildResult()
        if not isinstance(result, InfResult):
            raise TypeError("buildResult() must return InfResult.")
        return result

    def finalize(self):
        """
        Finalize the inference run and return the final result.
        """
        self.state.runtime = time.perf_counter() - self.startTime
        result = self.buildResult()
        Verbose.printConclusion(self, result)
        if self.logFlag:
            Verbose.saveLog(self)
        if self.saveFlag and self.storageCtx is not None:
            self.storage.saveSnapshot(self.storageCtx, self, result, isFinal=True)
            self.storage.saveResultArtifact(self.storageCtx, result)
            self.storage.close(self.storageCtx)
            self.storageCtx = None
        return result

    def evaluate(self, decs: np.ndarray):
        """
        Evaluate the problem and return internally oriented objectives.

        Args:
            decs: Decision matrix.

        Returns:
            Oriented objectives and constraints.
        """
        decs = self.problem.apply_var_type(decs)
        res = self.problem.evaluate(decs)
        self.FEs += decs.shape[0]
        return res.objs * self.problem.opt, res.cons

    def accept(self, objStar, objCur, consStar=None, qRatio=1.0, decStar=None, decCur=None, consCur=None):
        """
        Apply Metropolis acceptance with hard-constraint rejection.
        """
        if qRatio <= 0 or not np.isfinite(qRatio):
            return False
        logRatio = (
            self.log_prob(objStar, decs=decStar, cons=consStar)
            - self.log_prob(objCur, decs=decCur, cons=consCur)
            + np.log(qRatio)
        )
        feasible = True
        if self.problem.nCons > 0:
            feasible = np.all(np.asarray(consStar) <= 0)
        return bool(np.log(np.random.rand()) < float(np.ravel(logRatio)[0]) and feasible)

    def setProblem(self, problem: ProblemABC):
        """
        Set the problem instance for the inference run.

        Args:
            problem: Problem instance defining the inference space.
        """
        self.problem = problem

    def setParaVal(self, key, value):
        """
        Set an inference parameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        self.params.set(key, value)

    def getParaVal(self, *args):
        """
        Retrieve one or more inference parameters.

        Args:
            *args: Parameter names.

        Returns:
            The requested parameter value or values.
        """
        return self.params.get(*args)

    def log_prob(self, y, decs=None, cons=None):
        """
        Convert objective values into log probability.

        The default convention is `log_prob = -oriented_obj`. Users can provide
        `logProbFunc` to override this behavior.
        """
        if self.logProbFunc is not None:
            return np.asarray(self.logProbFunc(y, decs=decs, cons=cons))
        arr = np.asarray(y)
        if arr.ndim == 1:
            return -arr
        return -arr[..., 0]

    def _check_bound_(self, X, ub, lb):
        span = ub - lb
        y = (X - lb) % (2 * span)
        y = np.where(y > span, 2 * span - y, y)
        return lb + y

    def _check_gamma_(self, gamma):
        nChains = self.getParaVal("nChains")
        nInput = self.problem.nInput

        if isinstance(gamma, (float, int)):
            gamma = np.full((nChains, nInput), float(gamma))
        elif isinstance(gamma, list):
            gamma = np.asarray(gamma)

        if isinstance(gamma, np.ndarray):
            gamma = np.atleast_2d(gamma)
            n, _ = gamma.shape
            if n == 1:
                gamma = np.tile(gamma, (nChains, 1))
            elif n != nChains:
                raise ValueError("The shape of gamma must be (nChains, nInput) or (1, nInput)")
        else:
            raise ValueError("gamma must be a float, list, or numpy array")

        return gamma


class Params:
    """
    Helper container for inference parameters.
    """

    def __init__(self):
        """
        Initialize the parameter container.
        """
        self.data = {}

    @property
    def dicts(self):
        return self.data

    @property
    def keys(self):
        return list(self.data.keys())

    @property
    def values(self):
        return list(self.data.values())

    def items(self):
        return self.data.items()

    def asDict(self):
        return dict(self.data)

    def set(self, key, value):
        """
        Set a parameter value.
        """
        self.data[key] = value

    def get(self, *args):
        """
        Get one or more parameter values.

        Args:
            *args: Parameter names.

        Returns:
            The requested parameter value or values.
        """
        values = [self.data[arg] for arg in args]
        if len(args) > 1:
            return tuple(values)
        return values[0]

    def setPara(self, key, value):
        self.set(key, value)

    def getVal(self, *args):
        return self.get(*args)
