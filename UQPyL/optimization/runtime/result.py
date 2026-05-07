import json
from dataclasses import dataclass, field

import numpy as np

from ...core.runtime import export_runtime_meta

from ..metric import HV


@dataclass
class OptHistory:
    """
    Optimization history container.
    """
    populations: list = field(default_factory=list)
    bests: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    iterToFEs: list = field(default_factory=list)
    bestObjHistory: list = field(default_factory=list)
    numBestHistory: list = field(default_factory=list)
    bestMetricHistory: list = field(default_factory=list)
    improvedHistory: list = field(default_factory=list)

    def reset(self):
        self.populations.clear()
        self.bests.clear()
        self.metrics.clear()
        self.iterToFEs.clear()
        self.bestObjHistory.clear()
        self.numBestHistory.clear()
        self.bestMetricHistory.clear()
        self.improvedHistory.clear()

    def toDict(self) -> dict:
        return {
            "populations": list(self.populations),
            "bests": list(self.bests),
            "metrics": list(self.metrics),
            "iter_to_fes": [list(item) for item in self.iterToFEs],
            "best_obj_history": list(self.bestObjHistory),
            "num_best_history": list(self.numBestHistory),
            "best_metric_history": list(self.bestMetricHistory),
            "improved_history": list(self.improvedHistory),
        }


@dataclass
class OptResult:
    """
    Final optimization result.
    """
    bestDecs: np.ndarray | None
    bestObjs: np.ndarray | None
    bestCons: np.ndarray | None
    bestMetric: float | None
    bestFeasible: bool
    appearFEs: int | None
    appearIters: int | None
    FEs: int
    iters: int
    runtime: float
    history: OptHistory
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return export_runtime_meta(
            run_id=None,
            method=None,
            problem_name=None,
            n_input=None,
            n_output=None,
            n_con=None,
            runtime=float(self.runtime),
            created_at=None,
            extra={
                "best_feasible": bool(self.bestFeasible),
                "appear_fes": self.appearFEs,
                "appear_iters": self.appearIters,
                "fes": int(self.FEs),
                "iters": int(self.iters),
            },
        )

    def toDict(self) -> dict:
        return {
            **self.summary(),
            "best_decs": None if self.bestDecs is None else self.bestDecs.copy(),
            "best_objs": None if self.bestObjs is None else self.bestObjs.copy(),
            "best_cons": None if self.bestCons is None else self.bestCons.copy(),
            "best_metric": self.bestMetric,
            "history": self.history.toDict(),
            "extra": dict(self.extra),
        }


class OptState:
    """
    Mutable optimization state used during a run.
    """
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.history = OptHistory()
        self.reset()

    @property
    def bestObj(self):
        if self.bestObjs is None:
            return None
        return self.bestObjs[0, 0] if self.bestObjs.size != 0 else None

    def update(self, pop, problem, FEs, iters, algType):
        pop.requireEvaluated()

        self.currentPop = pop
        if algType == "EA":
            improved = self._updateSingle(pop, FEs, iters)
        else:
            improved = self._updateMulti(pop, FEs, iters)
        self._updateHistory(pop, FEs, iters, improved)

    def _updateSingle(self, pop, FEs, iters):
        bestPop = pop.getBest(k=1)
        localBestDecs = bestPop.decs
        localBestObjs = bestPop.objs
        localBestCons = bestPop.cons
        localBestFeasible = True if localBestCons is None else bool(np.all(np.maximum(0, localBestCons) <= 0))

        improved = False
        if self.bestObjs is None:
            improved = True
        elif localBestFeasible and not self.bestFeasible:
            improved = True
        elif localBestFeasible == self.bestFeasible and float(localBestObjs[0, 0]) < float(self.bestObjs[0, 0]):
            improved = True

        if improved:
            self.bestDecs = localBestDecs.copy()
            self.bestObjs = localBestObjs.copy()
            self.bestCons = None if localBestCons is None else localBestCons.copy()
            self.bestFeasible = localBestFeasible
            self.appearFEs = FEs
            self.appearIters = iters

        self.bestMetric = None
        return improved

    def _updateMulti(self, pop, FEs, iters):
        bestPop = pop.getBest()
        localBestDecs = bestPop.decs
        localBestObjs = bestPop.objs
        localBestCons = bestPop.cons
        localBestFeasible = True if localBestCons is None else bool(np.all(np.maximum(0, localBestCons) <= 0))
        refPoint = self._getHvRefPoint(localBestObjs)
        localMetric = float(HV(localBestObjs, refPoint=refPoint))

        improved = self.bestMetric is None or localMetric > self.bestMetric

        self.bestDecs = localBestDecs.copy()
        self.bestObjs = localBestObjs.copy()
        self.bestCons = None if localBestCons is None else localBestCons.copy()
        self.bestFeasible = localBestFeasible
        self.bestMetric = localMetric

        if improved:
            self.appearFEs = FEs
            self.appearIters = iters

        return improved

    def _updateHistory(self, pop, FEs, iters, improved):
        self.history.populations.append(
            {
                "decs": pop.decs.copy(),
                "objs": None if pop.objs is None else pop.objs.copy(),
                "cons": None if pop.cons is None else pop.cons.copy(),
            }
        )
        self.history.bests.append(
            {
                "bestDecs": None if self.bestDecs is None else self.bestDecs.copy(),
                "bestObjs": None if self.bestObjs is None else self.bestObjs.copy(),
                "bestCons": None if self.bestCons is None else self.bestCons.copy(),
            }
        )
        self.history.metrics.append(self.bestMetric)
        self.history.iterToFEs.append([iters, FEs])
        self.history.improvedHistory.append(bool(improved))

        if self.bestObjs is not None and self.bestObjs.shape[0] == 1:
            self.history.bestObjHistory.append(float(self.bestObjs[0, 0]))
        elif self.bestObjs is not None:
            self.history.numBestHistory.append(int(self.bestObjs.shape[0]))
            self.history.bestMetricHistory.append(self.bestMetric)

    def buildResult(self):
        return OptResult(
            bestDecs=None if self.bestDecs is None else self.bestDecs.copy(),
            bestObjs=None if self.bestObjs is None else self.bestObjs.copy(),
            bestCons=None if self.bestCons is None else self.bestCons.copy(),
            bestMetric=self.bestMetric,
            bestFeasible=self.bestFeasible,
            appearFEs=self.appearFEs,
            appearIters=self.appearIters,
            FEs=self.algorithm.FEs,
            iters=self.algorithm.iters,
            runtime=self.runtime,
            history=self.history,
            extra=self.extra.copy(),
        )

    def toNpzPayload(self):
        result = self.buildResult()
        problem = self.algorithm.problem
        summary = {
            "algorithm": self.algorithm.name,
            "problem": getattr(problem, "name", problem.__class__.__name__),
            "nInput": int(problem.nInput),
            "nObj": int(problem.nObj),
            "nCon": int(problem.nCon),
            "FEs": int(result.FEs),
            "iters": int(result.iters),
            "runtime": float(result.runtime),
            "bestFeasible": bool(result.bestFeasible),
            "bestMetric": None if result.bestMetric is None else float(result.bestMetric),
            "appearFEs": result.appearFEs,
            "appearIters": result.appearIters,
        }

        payload = {
            "bestDecs": np.empty((0, 0)) if result.bestDecs is None else result.bestDecs,
            "bestObjs": np.empty((0, 0)) if result.bestObjs is None else result.bestObjs,
            "iterToFEs": np.asarray(self.history.iterToFEs, dtype=np.int64),
            "summaryJson": np.array(json.dumps(summary), dtype="<U4096"),
        }
        if result.bestCons is not None:
            payload["bestCons"] = result.bestCons
        if self.history.bestObjHistory:
            payload["bestObjHistory"] = np.asarray(self.history.bestObjHistory, dtype=float)
        if self.history.numBestHistory:
            payload["numBestHistory"] = np.asarray(self.history.numBestHistory, dtype=np.int64)
        if self.history.bestMetricHistory:
            payload["bestMetricHistory"] = np.asarray(self.history.bestMetricHistory, dtype=float)
        return payload

    def reset(self):
        self.bestDecs = None
        self.bestObjs = None
        self.bestCons = None
        self.bestMetric = None
        self.bestFeasible = False
        self.appearFEs = None
        self.appearIters = None
        self.currentPop = None
        self.hvRefPoint = None
        self.runtime = 0.0
        self.extra = {}
        self.history.reset()

    def _getHvRefPoint(self, bestObjs):
        if self.hvRefPoint is not None:
            return self.hvRefPoint

        algRefPoint = getattr(self.algorithm, "hvRefPoint", None)
        if algRefPoint is not None:
            self.hvRefPoint = np.asarray(algRefPoint, dtype=float).copy()
            return self.hvRefPoint

        worst = np.max(np.asarray(bestObjs, dtype=float), axis=0)
        self.hvRefPoint = np.where(worst == 0.0, 0.2, worst * 1.2)
        return self.hvRefPoint


Result = OptState
