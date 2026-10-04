import abc
import os
import time

import numpy as np

from .chain import Chain
from .runtime import InfResult, Result, SqliteStorage, Verbose
from ..doe import LHS
from ..core import config
from ..core.params import Params
from ..core.runtime_session import RunSession
from ..core.runtime_lifecycle import RunLifecycle
from ..problem import ProblemABC


class InferenceABC(RunLifecycle, metaclass=abc.ABCMeta):
    """
    Abstract base class for inference methods.
    Shared workflow and utilities for MCMC-style sampling methods.
    """

    minChains = 1
    boundaryPolicy = "reflect"
    updateMode = "independent"
    adaptationPhase = "none"
    proposalFamily = "random_walk"

    @classmethod
    def getCapabilities(cls):
        return {
            "min_objectives": 1,
            "max_objectives": 1,
            "constraint_handling": "hard_rejection",
            "variable_types": ["continuous", "integer", "discrete"],
            "requires_finite_bounds": True,
        }

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
            maxInitAttempts: Maximum LHS batches used to find feasible initial chains with positive probability.
        """
        # Initialize settings and results
        self.params = Params()
        self.set("nChains", 1)
        self.set("warmUp", 0)
        self.state = Result(self)

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
        self.runId = None
        self.session: RunSession | None = None
        self.set("maxInitAttempts", maxInitAttempts)
        if logProbFunc is not None:
            self.set("logProbFunc", getattr(logProbFunc, "__name__", repr(logProbFunc)))

    @abc.abstractmethod
    def run(self, problem=None, *args, **kwargs):
        """
        Run the inference workflow.
        """
        raise NotImplementedError

    def setup(self, problem: ProblemABC, seed: int = None):
        """
        Set the problem, reset runtime state, and initialize optional storage.

        Args:
            problem: Problem instance defining the inference space.
            seed: Optional random seed.
        """
        self.setProblem(problem)
        self._startRun()
        self.reset()
        self.validateParameters()
        self.validateProblem()
        Verbose.setupContext(self, problem)

        if self.saveFlag:
            rootDir = config.resolveWorkDir(getattr(problem, "workDir", None))
            self.storage = SqliteStorage(rootDir)

        # Initialize random seed
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 1000000))
        self.rng = np.random.default_rng(seed)

        self.set("seed", seed)
        self.set("saveFreq", self.saveFreq)

        if self.saveFlag:
            self.session = self.storage.create_run(self)
            self.runId = self.session.run_id

        Verbose.printSettings(self)

    @staticmethod
    def validateInteger(name, value, minimum):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    def validateParameters(self):
        """Validate shared settings before opening storage or evaluating a model."""
        for name, value, minimum in [
            ("nChains", self.get("nChains"), self.minChains),
            ("warmUp", self.get("warmUp"), 0),
            ("maxIters", self.maxIters, 1),
            ("maxInitAttempts", self.get("maxInitAttempts"), 1),
            ("verboseFreq", self.verboseFreq, 1),
            ("saveFreq", self.saveFreq, 1),
        ]:
            self.validateInteger(name, value, minimum)
        distribution = self.params.data.get("propDist")
        if "propDist" in self.params.data and distribution not in ("gauss", "uniform"):
            raise ValueError("propDist must be 'gauss' or 'uniform'.")
        if self.logProbFunc is not None and not callable(self.logProbFunc):
            raise ValueError("logProbFunc must be callable or None.")

    def setSamplerDiagnostics(self, gamma, proposalSettings=None, **details):
        """Publish shared policy fields without changing sampling or RNG state."""
        settings = {
            "gamma": None if gamma is None else np.asarray(gamma).tolist(),
            "distribution": self.params.data.get("propDist"),
        }
        settings.update(proposalSettings or {})
        self.state.diagnostics["sampler"] = {
            "boundary_policy": self.boundaryPolicy,
            "update_mode": self.updateMode,
            "adaptation_phase": self.adaptationPhase,
            "proposal_family": self.proposalFamily,
            "proposal_settings": settings,
            **details,
        }

    def validateProblem(self):
        """
        Validate problem-level assumptions shared by inference algorithms.
        """
        if self.problem.nOutput != 1:
            raise ValueError("Inference currently supports scalar objectives only.")
        self.problem.unit_to_space(np.full((1, self.problem.nInput), 0.5))
        for idx in self.problem.space.idxD:
            if self.problem.ub[0, idx] == self.problem.lb[0, idx] and len(self.problem.space.varSet[idx]) > 1:
                raise ValueError("Discrete inference with multiple choices requires a positive latent interval.")

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
        Generate internal latent samples for all chains.

        Evaluate decoded real values, retaining feasible initial states with
        finite log probability. Do not use returned latent decisions as
        standalone model inputs; public results contain decoded decisions.
        """
        sampler = LHS()
        xs, objectives, constraints = [], [], []
        maxAttempts = self.get("maxInitAttempts")
        for attempt in range(maxAttempts):
            sampleSeed = (
                seed if attempt == 0 and problem.nCons == 0 and seed is not None else int(self.rng.integers(0, 1000000))
            )
            batch = self._sampleLatent(sampler, nChains, sampleSeed)
            objs, cons = self.evaluate(batch)
            valid = np.isfinite(self.log_prob(objs, decs=batch, cons=cons))
            if problem.nCons > 0:
                if cons is None:
                    raise ValueError("Constrained inference problem must return constraint values.")
                valid &= (cons <= 0).all(axis=1)
            if attempt == 0 and valid.all():
                return batch, objs, cons
            for index in np.flatnonzero(valid):
                xs.append(batch[index])
                objectives.append(objs[index])
                if problem.nCons > 0:
                    constraints.append(cons[index])
                if len(xs) == nChains:
                    return (
                        np.asarray(xs),
                        np.asarray(objectives),
                        np.asarray(constraints) if problem.nCons > 0 else None,
                    )
        raise ValueError(
            f"Unable to initialize {nChains} feasible positive-probability chains after {maxAttempts} LHS batches."
        )

    def _sampleLatent(self, sampler, nSamples, seed):
        unit = sampler.sample(self.problem, nSamples, seed, output="unit")
        return self.problem.lb + unit * (self.problem.ub - self.problem.lb)

    def _decodeDecs(self, decs):
        """Decode latent bounded coordinates without modifying the Markov state.

        Continuous axes retain physical units. Integer/discrete choices occupy
        equal-width intervals so a flat target assigns equal mass to each choice.
        """
        original = np.asarray(decs)
        values = np.asarray(self.problem.validate(decs), dtype=float).copy()
        if getattr(self.problem.space, "encoding", "real") == "mix":
            span = self.problem.ub - self.problem.lb
            unit = np.divide(values - self.problem.lb, span, out=np.full_like(values, 0.5), where=span > 0)
            decoded = self.problem.unit_to_space(unit)
            # Avoid even a round-trip change to continuous coordinates.
            decoded[:, self.problem.space.idxF] = values[:, self.problem.space.idxF]
            values = decoded
        return values[0] if original.ndim == 1 else values

    def initChains(self, nChains: int, X: np.ndarray, objs: np.ndarray, cons: np.ndarray = None):
        """
        Initialize chain containers from current states.
        """
        nInput = self.problem.nInput
        nOutput = self.problem.nOutput
        nCons = self.problem.nCons
        chains = [Chain(nInput, nOutput, nCons, self.maxIters) for _ in range(nChains)]
        logProb = self.log_prob(objs, decs=X, cons=cons)
        if not np.all(np.isfinite(logProb)):
            raise ValueError("Initial chain states must have finite log probability.")

        for i, chain in enumerate(chains):
            chain.add(
                X[i],
                objs[i],
                cons[i] if nCons > 0 else None,
                logProb=logProb[i],
                accepted=True,
            )

        return chains

    def updateChainState(self, index, current, currentObjs, currentCons, decision, objective, constraints):
        """Commit one accepted row; callers retain control of transition ordering."""
        current[index] = decision
        currentObjs[index] = objective
        if self.problem.nCons > 0:
            currentCons[index] = constraints

    def recordChainState(self, chain, index, current, currentObjs, currentCons, accepted):
        """Append the occupied state, including rejected repeats, in caller order."""
        constraints = currentCons[index] if self.problem.nCons > 0 else None
        chain.add(
            current[index],
            currentObjs[index],
            constraints,
            logProb=self.log_prob(currentObjs[index], decs=current[index], cons=constraints),
            accepted=accepted,
        )

    def update(self, chains):
        """
        Update runtime state from chains and handle progress output.
        """
        self.state.runtime = time.perf_counter() - self.startTime
        self.state.update(chains, self.problem, self.FEs, self.iters)
        if self.verboseFlag or self.logFlag:
            Verbose.printIteration(self)
        if self.saveFlag and self.session is not None and self.iters % self.saveFreq == 0:
            self.storage.saveSnapshot(self.session, self, isFinal=False)
        return self.state

    def checkTermination(self, chains=None):
        """
        Advance the sampling iteration counter.
        """
        if self.iters >= self.maxIters - 1:
            self.state.stopReason = "max_iters"
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
        if self.state.stopReason is None:
            self.state.stopReason = "completed"
        if self.state.history.snapshots:
            self.state.history.snapshots[-1]["stop_reason"] = self.state.stopReason
        self.state.runtime = time.perf_counter() - self.startTime
        result = self.buildResult()
        Verbose.printConclusion(self, result)
        if self.logFlag:
            Verbose.saveLog(self)
        if self.saveFlag and self.session is not None:
            self.storage.saveSnapshot(self.session, self, result, isFinal=True)
            self.storage.saveResultArtifact(self.session, result)
            self._closeStandaloneSession()
        return result

    def evaluate(self, decs: np.ndarray):
        """
        Evaluate the problem and return internally oriented objectives.

        Args:
            decs: Internal latent decision matrix in the problem's bounded axes.

        Returns:
            Oriented objectives and constraints.
        """
        realDecs = np.atleast_2d(self._decodeDecs(decs))
        res = self.problem.evaluate(realDecs)
        self.FEs += realDecs.shape[0]
        return res.objs * self.problem.opt, res.cons

    def evaluateProposal(self, proposed, currentObjs, currentCons):
        """Reject raw out-of-box proposals without evaluating the user model.

        Reflection is not symmetric for general correlated/directional kernels.
        Cached values are retained for rejected rows; callers must also use the
        returned admissibility mask when recording acceptance.
        """
        admissible = np.all(
            np.isfinite(proposed) & (proposed >= self.problem.lb) & (proposed <= self.problem.ub), axis=1
        )
        objs = currentObjs.copy()
        cons = None if currentCons is None else currentCons.copy()
        if np.any(admissible):
            evaluatedObjs, evaluatedCons = self.evaluate(proposed[admissible])
            objs[admissible] = evaluatedObjs
            if self.problem.nCons > 0:
                cons[admissible] = evaluatedCons
        return objs, cons, admissible

    def accept(
        self, objStar, objCur, consStar=None, qRatio=1.0, decStar=None, decCur=None, consCur=None, logQRatio=None
    ):
        """
        Apply Metropolis acceptance with hard-constraint rejection.
        """
        if logQRatio is None:
            if qRatio <= 0 or not np.isfinite(qRatio):
                return False
            logQRatio = np.log(qRatio)
        elif np.isnan(logQRatio):
            return False
        proposedLog = float(self.log_prob(objStar, decs=decStar, cons=consStar)[0])
        currentLog = float(self.log_prob(objCur, decs=decCur, cons=consCur)[0])
        # A zero-density proposal is always rejected, including -inf -> -inf.
        # Initial states are in support, but allow recovery in direct helper use.
        if proposedLog == -np.inf or logQRatio == -np.inf:
            return False
        logRatio = proposedLog - currentLog + logQRatio
        feasible = True
        if self.problem.nCons > 0:
            feasible = np.all(np.asarray(consStar) <= 0)
        uniform = self.rng.random()
        logUniform = -np.inf if uniform == 0 else np.log(uniform)
        return bool(logUniform < logRatio and feasible)

    def setProblem(self, problem: ProblemABC):
        """
        Set the problem instance for the inference run.

        Args:
            problem: Problem instance defining the inference space.
        """
        self.problem = problem

    def set(self, key, value):
        """
        Set an inference parameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        self.params.set(key, value)

    def get(self, *args):
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
        objectives = np.asarray(y)
        count = objectives.shape[0] if objectives.ndim > 1 else 1
        if self.logProbFunc is not None:
            realDecs = None if decs is None else self._decodeDecs(decs)
            values = np.asarray(self.logProbFunc(y, decs=realDecs, cons=cons))
        else:
            values = -objectives if objectives.ndim <= 1 else -objectives[..., 0]
        validShape = values.shape in [(count,), (count, 1)] or (count == 1 and values.ndim == 0)
        if not validShape:
            raise ValueError(f"log probability must return one value per row: expected ({count},) or ({count}, 1).")
        if values.dtype.kind not in "iuf":
            raise ValueError("log probability must contain real numeric values.")
        values = values.reshape(count).astype(float, copy=False)
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise ValueError("log probability contains NaN or +inf; use finite values or -inf for zero probability.")
        return values

    def _check_bound_(self, X, ub, lb):
        span = ub - lb
        safeSpan = np.where(span > 0, span, 1.0)
        y = (X - lb) % (2 * safeSpan)
        y = np.where(y > safeSpan, 2 * safeSpan - y, y)
        return np.where(span > 0, lb + y, lb)

    def _check_gamma_(self, gamma):
        nChains = self.get("nChains")
        nInput = self.problem.nInput
        try:
            values = np.asarray(gamma)
        except (TypeError, ValueError) as error:
            raise ValueError("gamma must be a real scalar, vector or matrix.") from error
        if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("gamma must contain finite nonnegative real values.")
        if values.ndim == 0:
            return np.full((nChains, nInput), float(values))
        values = np.atleast_2d(values)
        if values.shape == (1, nInput):
            return np.tile(values, (nChains, 1))
        if values.shape == (nChains, nInput):
            return values.copy()
        raise ValueError("gamma must have shape (nInput,), (1, nInput) or (nChains, nInput).")
