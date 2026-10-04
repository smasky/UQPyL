from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime
from typing import Any

import numpy as np

from ...core.runtime import export_runtime_meta


@dataclass
class InfHistory:
    snapshots: list = field(default_factory=list)
    iterToFEs: list = field(default_factory=list)
    meanLogProbHistory: list = field(default_factory=list)
    acceptanceRateHistory: list = field(default_factory=list)
    feasibleRateHistory: list = field(default_factory=list)
    bestObjHistory: list = field(default_factory=list)

    def reset(self):
        self.snapshots.clear()
        self.iterToFEs.clear()
        self.meanLogProbHistory.clear()
        self.acceptanceRateHistory.clear()
        self.feasibleRateHistory.clear()
        self.bestObjHistory.clear()


@dataclass
class InfResult:
    runId: str | None
    method: str
    problemName: str
    nInput: int
    nOutput: int
    nCon: int
    settings: dict[str, Any]
    runtime: float
    createdAt: str
    decs: np.ndarray
    objs: np.ndarray
    cons: np.ndarray | None
    logProb: np.ndarray
    accepted: np.ndarray
    feasibleMask: np.ndarray
    acceptanceRate: np.ndarray
    bestDecs: np.ndarray | None
    bestObjs: np.ndarray | None
    bestCons: np.ndarray | None
    bestFeasible: bool
    FEs: int
    iters: int
    history: InfHistory
    diagnostics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    stopReason: str | None = None

    def computeDiagnostics(self):
        """Compute post-run chain diagnostics and return an independent report.

        The complete result remains unchanged except for its diagnostics field.
        No model evaluations, RNG draws, or automatic stopping are performed.
        """
        from ..diagnostics import computeChainDiagnostics

        report = computeChainDiagnostics(self.decs)
        self.diagnostics["chains"] = report
        return deepcopy(report)

    def summary(self) -> dict[str, Any]:
        return export_runtime_meta(
            run_id=self.runId,
            method=self.method,
            problem_name=self.problemName,
            n_input=self.nInput,
            n_output=self.nOutput,
            n_con=self.nCon,
            runtime=self.runtime,
            created_at=self.createdAt,
            extra={
                "n_chains": int(self.decs.shape[0]),
                "draws": int(self.decs.shape[1]),
                "fes": self.FEs,
                "iters": self.iters,
                "stop_reason": self.stopReason,
                "acceptance_rate_mean": float(np.mean(self.acceptanceRate)) if self.acceptanceRate.size else 0.0,
                "feasible_rate": float(np.mean(self.feasibleMask)) if self.feasibleMask.size else 0.0,
                "best_feasible": self.bestFeasible,
            },
        )

    def toDict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "settings": deepcopy(self.settings),
            "decs": self.decs.copy(),
            "objs": self.objs.copy(),
            "cons": None if self.cons is None else self.cons.copy(),
            "log_prob": self.logProb.copy(),
            "accepted": self.accepted.copy(),
            "feasible_mask": self.feasibleMask.copy(),
            "acceptance_rate": self.acceptanceRate.copy(),
            "best_decs": None if self.bestDecs is None else self.bestDecs.copy(),
            "best_objs": None if self.bestObjs is None else self.bestObjs.copy(),
            "best_cons": None if self.bestCons is None else self.bestCons.copy(),
            "diagnostics": deepcopy(self.diagnostics),
            "extra": deepcopy(self.extra),
        }


class InfState:
    def __init__(self, inference):
        self.inference = inference
        self.history = InfHistory()
        self.reset()

    def reset(self):
        self._resetSamples()
        self.stopReason = None
        self.runtime = 0.0
        self.createdAt = datetime.now().isoformat(timespec="seconds")
        self.diagnostics = {"chains": {"status": "not_computed"}}
        self.extra = {}
        self.history.reset()

    def _resetSamples(self):
        self._chains = ()
        self._draws = 0
        self._buffers = {}
        self._logProbSum = 0.0
        self._logProbCount = 0
        self._feasibleCount = 0
        self._allMoments = (0, None, None)
        self._feasibleMoments = (0, None, None)
        self.decs = None
        self.objs = None
        self.cons = None
        self.logProb = None
        self.accepted = None
        self.feasibleMask = None
        self.acceptanceRate = None
        self.bestDecs = None
        self.bestObjs = None
        self.bestCons = None
        self.bestFeasible = False

    def update(self, chains, problem, FEs, iters):
        self._collect(chains, problem)
        snapshot = self.buildSnapshot(FEs, iters)
        self.history.snapshots.append(snapshot)
        self.history.iterToFEs.append([iters, FEs])
        self.history.meanLogProbHistory.append(snapshot["meanLogProb"])
        self.history.acceptanceRateHistory.append(snapshot["acceptanceRateMean"])
        self.history.feasibleRateHistory.append(snapshot["feasibleRate"])
        if snapshot["bestObj"] is not None:
            self.history.bestObjHistory.append(snapshot["bestObj"])
        return self

    def buildSnapshot(self, FEs, iters):
        return {
            "iter": int(iters),
            "stop_reason": self.stopReason,
            "FEs": int(FEs),
            "runtime": float(self.runtime),
            "meanLogProb": self.meanLogProb,
            "bestObj": self.bestObj,
            "feasibleRate": self.feasibleRate,
            "acceptanceRateMean": self.acceptanceRateMean,
        }

    def buildResult(self):
        problem = self.inference.problem
        session = getattr(self.inference, "session", None)
        runId = None if session is None else getattr(session, "run_id", None)
        return InfResult(
            stopReason=self.stopReason,
            runId=runId if runId is not None else getattr(self.inference, "runId", None),
            method=self.inference.name,
            problemName=problem.name,
            nInput=problem.nInput,
            nOutput=problem.nOutput,
            nCon=problem.nCons,
            settings=deepcopy(self.inference.params.asDict()),
            runtime=float(self.runtime),
            createdAt=self.createdAt,
            decs=np.empty((0, 0, 0)) if self.decs is None else self.decs.copy(),
            objs=np.empty((0, 0, 0)) if self.objs is None else self.objs * problem.opt,
            cons=None if self.cons is None else self.cons.copy(),
            logProb=np.empty((0, 0)) if self.logProb is None else self.logProb.copy(),
            accepted=np.empty((0, 0), dtype=bool) if self.accepted is None else self.accepted.copy(),
            feasibleMask=np.empty((0, 0), dtype=bool) if self.feasibleMask is None else self.feasibleMask.copy(),
            acceptanceRate=np.empty((0,)) if self.acceptanceRate is None else self.acceptanceRate.copy(),
            bestDecs=None if self.bestDecs is None else self.bestDecs.copy(),
            bestObjs=None if self.bestObjs is None else self.bestObjs.copy(),
            bestCons=None if self.bestCons is None else self.bestCons.copy(),
            bestFeasible=bool(self.bestFeasible),
            FEs=self.inference.FEs,
            iters=self.inference.iters,
            history=deepcopy(self.history),
            diagnostics=deepcopy(self.diagnostics),
            extra=deepcopy(self.extra),
        )

    @property
    def meanLogProb(self):
        if self.logProb is None or self.logProb.size == 0:
            return None
        return self._logProbSum / self._logProbCount if self._logProbCount else float("nan")

    @property
    def bestObj(self):
        if self.bestObjs is None:
            return None
        return float(np.ravel(self.bestObjs)[0])

    @property
    def feasibleRate(self):
        if self.feasibleMask is None or self.feasibleMask.size == 0:
            return 0.0
        return self._feasibleCount / self.feasibleMask.size

    @property
    def acceptanceRateMean(self):
        if self.acceptanceRate is None or self.acceptanceRate.size == 0:
            return 0.0
        return float(np.mean(self.acceptanceRate))

    def _collect(self, chains, problem):
        """Append only newly completed draws; chains remain the sampler's storage."""
        draw = min((chain.count for chain in chains), default=0)
        if (
            len(chains) != len(self._chains)
            or draw < self._draws
            or any(new is not old for new, old in zip(chains, self._chains))
        ):
            self._resetSamples()
        if draw == self._draws:
            return
        if not self._buffers:
            self._chains = tuple(chains)
            capacity = min(len(chain.decs) for chain in chains)
            shape = (len(chains), capacity)
            for name, tail, dtype in (
                ("decs", (problem.nInput,), float),
                ("objs", (problem.nOutput,), float),
                ("logProb", (), float),
                ("accepted", (), bool),
                ("feasibleMask", (), bool),
            ):
                self._buffers[name] = np.empty(shape + tail, dtype=dtype)
            if problem.nCons:
                self._buffers["cons"] = np.empty(shape + (problem.nCons,))
            self._acceptedCounts = np.zeros(len(chains), dtype=np.int64)
            self._bestIndices = np.full(len(chains), -1, dtype=int)

        start = self._draws
        latent = np.stack([chain.decs[start:draw] for chain in chains])
        self._buffers["decs"][:, start:draw] = self.inference._decodeDecs(latent.reshape(-1, problem.nInput)).reshape(
            latent.shape
        )
        for name in ("objs", "logProb", "accepted", "cons"):
            if name in self._buffers:
                self._buffers[name][:, start:draw] = np.stack([getattr(chain, name)[start:draw] for chain in chains])
        feasible = (
            np.all(self._buffers["cons"][:, start:draw] <= 0, axis=2)
            if problem.nCons
            else np.ones((len(chains), draw - start), dtype=bool)
        )
        self._buffers["feasibleMask"][:, start:draw] = feasible
        for name, buffer in self._buffers.items():
            setattr(self, name, buffer[:, :draw])
        self._draws = draw

        self._acceptedCounts += np.count_nonzero(self.accepted[:, max(start, 1) : draw], axis=1)
        self.acceptanceRate = self._acceptedCounts / (draw - 1) if draw > 1 else np.ones(len(chains))
        self._feasibleCount += int(np.count_nonzero(feasible))
        logProb = self.logProb[:, start:draw]
        with np.errstate(invalid="ignore"):
            self._logProbSum += float(np.nansum(logProb))
        self._logProbCount += int(np.count_nonzero(~np.isnan(logProb)))
        values = self.decs[:, start:draw].reshape(-1, problem.nInput)
        self._allMoments = self._mergeMoments(self._allMoments, values)
        self._feasibleMoments = self._mergeMoments(self._feasibleMoments, values[feasible.ravel()])
        self._updateBest(problem, start)

    @staticmethod
    def _mergeMoments(current, values):
        count, mean, m2 = current
        if not len(values):
            return current
        batchMean = np.mean(values, axis=0)
        batchM2 = np.sum((values - batchMean) ** 2, axis=0)
        if not count:
            return len(values), batchMean, batchM2
        total = count + len(values)
        delta = batchMean - mean
        return (total, mean + delta * (len(values) / total), m2 + batchM2 + delta**2 * (count * len(values) / total))

    def decisionMoments(self):
        """Feasible-sample moments, falling back to all samples when none are feasible."""
        count, mean, m2 = self._feasibleMoments if self._feasibleCount else self._allMoments
        if not count:
            return None, None
        return mean.copy(), np.sqrt(np.maximum(m2 / count, 0))

    def _updateBest(self, problem, start):
        # Keep one incumbent per chain to preserve chain-major, earliest-draw ties.
        for chain in range(len(self._chains)):
            old = self._bestIndices[chain]
            feasible = self.feasibleMask[chain, start:]
            indices = np.flatnonzero(feasible) + start if np.any(feasible) else np.arange(start, self._draws)
            index = indices[np.argmin(self.objs[chain, indices, 0])]
            newFeasible = self.feasibleMask[chain, index]
            if (
                old < 0
                or (newFeasible and not self.feasibleMask[chain, old])
                or (
                    newFeasible == self.feasibleMask[chain, old]
                    and self.objs[chain, index, 0] < self.objs[chain, old, 0]
                )
            ):
                self._bestIndices[chain] = index
        chainIds = np.arange(len(self._chains))
        feasible = self.feasibleMask[chainIds, self._bestIndices]
        eligible = chainIds[feasible] if np.any(feasible) else chainIds
        chain = eligible[np.argmin(self.objs[eligible, self._bestIndices[eligible], 0])]
        index = self._bestIndices[chain]
        self.bestDecs = self.decs[chain, index : index + 1].copy()
        self.bestObjs = self.objs[chain, index : index + 1] * problem.opt
        self.bestCons = None if self.cons is None else self.cons[chain, index : index + 1].copy()
        self.bestFeasible = bool(feasible[chain])


Result = InfState
