from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


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

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "runId": self.runId,
            "problemName": self.problemName,
            "nInput": self.nInput,
            "nOutput": self.nOutput,
            "nCon": self.nCon,
            "nChains": int(self.decs.shape[0]),
            "draws": int(self.decs.shape[1]),
            "FEs": self.FEs,
            "iters": self.iters,
            "acceptanceRateMean": float(np.mean(self.acceptanceRate)) if self.acceptanceRate.size else 0.0,
            "feasibleRate": float(np.mean(self.feasibleMask)) if self.feasibleMask.size else 0.0,
            "bestFeasible": self.bestFeasible,
            "runtime": self.runtime,
            "createdAt": self.createdAt,
        }

    def toDict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "settings": dict(self.settings),
            "decs": self.decs.copy(),
            "objs": self.objs.copy(),
            "cons": None if self.cons is None else self.cons.copy(),
            "logProb": self.logProb.copy(),
            "accepted": self.accepted.copy(),
            "feasibleMask": self.feasibleMask.copy(),
            "acceptanceRate": self.acceptanceRate.copy(),
            "bestDecs": None if self.bestDecs is None else self.bestDecs.copy(),
            "bestObjs": None if self.bestObjs is None else self.bestObjs.copy(),
            "bestCons": None if self.bestCons is None else self.bestCons.copy(),
            "diagnostics": dict(self.diagnostics),
            "extra": dict(self.extra),
        }


class InfState:
    def __init__(self, inference):
        self.inference = inference
        self.history = InfHistory()
        self.reset()

    def reset(self):
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
        self.runtime = 0.0
        self.createdAt = datetime.now().isoformat(timespec="seconds")
        self.diagnostics = {}
        self.extra = {}
        self.history.reset()

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
            "FEs": int(FEs),
            "runtime": float(self.runtime),
            "meanLogProb": self.meanLogProb,
            "bestObj": self.bestObj,
            "feasibleRate": self.feasibleRate,
            "acceptanceRateMean": self.acceptanceRateMean,
        }

    def buildResult(self):
        problem = self.inference.problem
        return InfResult(
            runId=getattr(self.inference, "runId", None),
            method=self.inference.name,
            problemName=problem.name,
            nInput=problem.nInput,
            nOutput=problem.nOutput,
            nCon=problem.nCons,
            settings=self.inference.params.asDict(),
            runtime=float(self.runtime),
            createdAt=self.createdAt,
            decs=np.empty((0, 0, 0)) if self.decs is None else self.decs.copy(),
            objs=np.empty((0, 0, 0)) if self.objs is None else self.objs.copy(),
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
            history=self.history,
            diagnostics=dict(self.diagnostics),
            extra=dict(self.extra),
        )

    @property
    def meanLogProb(self):
        if self.logProb is None or self.logProb.size == 0:
            return None
        return float(np.nanmean(self.logProb))

    @property
    def bestObj(self):
        if self.bestObjs is None:
            return None
        return float(np.ravel(self.bestObjs)[0])

    @property
    def feasibleRate(self):
        if self.feasibleMask is None or self.feasibleMask.size == 0:
            return 0.0
        return float(np.mean(self.feasibleMask))

    @property
    def acceptanceRateMean(self):
        if self.acceptanceRate is None or self.acceptanceRate.size == 0:
            return 0.0
        return float(np.mean(self.acceptanceRate))

    def _collect(self, chains, problem):
        counts = [chain.count for chain in chains]
        draw = min(counts) if counts else 0
        if draw == 0:
            return

        self.decs = np.stack([chain.decs[:draw].copy() for chain in chains])
        self.objs = np.stack([chain.objs[:draw].copy() for chain in chains])
        self.logProb = np.stack([chain.logProb[:draw].copy() for chain in chains])
        self.accepted = np.stack([chain.accepted[:draw].copy() for chain in chains])

        if problem.nCons > 0:
            self.cons = np.stack([chain.cons[:draw].copy() for chain in chains])
            self.feasibleMask = (self.cons <= 0).all(axis=2)
        else:
            self.cons = None
            self.feasibleMask = np.ones((len(chains), draw), dtype=bool)

        self.acceptanceRate = np.mean(self.accepted[:, 1:], axis=1) if draw > 1 else np.ones(len(chains))
        self._updateBest(problem)

    def _updateBest(self, problem):
        flatObjs = self.objs.reshape(-1, self.objs.shape[-1])
        flatDecs = self.decs.reshape(-1, self.decs.shape[-1])
        flatCons = None if self.cons is None else self.cons.reshape(-1, self.cons.shape[-1])
        flatFeasible = self.feasibleMask.reshape(-1)

        if np.any(flatFeasible):
            feasibleObjs = flatObjs[flatFeasible]
            feasibleDecs = flatDecs[flatFeasible]
            feasibleCons = None if flatCons is None else flatCons[flatFeasible]
            idx = int(np.argmin(feasibleObjs[:, 0]))
            self.bestDecs = feasibleDecs[idx:idx + 1].copy()
            self.bestObjs = (feasibleObjs[idx:idx + 1] * problem.opt).copy()
            self.bestCons = None if feasibleCons is None else feasibleCons[idx:idx + 1].copy()
            self.bestFeasible = True
            return

        idx = int(np.argmin(flatObjs[:, 0]))
        self.bestDecs = flatDecs[idx:idx + 1].copy()
        self.bestObjs = (flatObjs[idx:idx + 1] * problem.opt).copy()
        self.bestCons = None if flatCons is None else flatCons[idx:idx + 1].copy()
        self.bestFeasible = False


Result = InfState
