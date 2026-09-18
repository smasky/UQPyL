import json
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from ...core.runtime import export_runtime_meta

from ..metric import HV
from ..core.constraint import compareSolutions, calcConstraintViolation


@dataclass
class OptHistory:
    """
    Optimization history container.
    """
    populations: list = field(default_factory=list)
    bests: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    iterToFEs: list = field(default_factory=list)
    snapshotIterToFEs: list = field(default_factory=list)
    bestObjHistory: list = field(default_factory=list)
    numBestHistory: list = field(default_factory=list)
    bestMetricHistory: list = field(default_factory=list)
    improvedHistory: list = field(default_factory=list)

    def reset(self):
        self.populations.clear()
        self.bests.clear()
        self.metrics.clear()
        self.iterToFEs.clear()
        self.snapshotIterToFEs.clear()
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
            "snapshot_iter_to_fes": [list(item) for item in self.snapshotIterToFEs],
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
    candidateDecs: np.ndarray | None = None
    candidateObjs: np.ndarray | None = None
    candidateCons: np.ndarray | None = None
    minViolation: float | None = None

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
            "candidate_decs": None if self.candidateDecs is None else self.candidateDecs.copy(),
            "candidate_objs": None if self.candidateObjs is None else self.candidateObjs.copy(),
            "candidate_cons": None if self.candidateCons is None else self.candidateCons.copy(),
            "min_violation": self.minViolation,
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
        self.extra["constraint_weights"] = None if pop.conWgt is None else pop.conWgt.copy()
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
        localViolation = calcConstraintViolation(localBestCons, pop.conWgt)
        localBestFeasible = localViolation is None or bool(localViolation[0] <= 0)

        # Reuse the same feasibility-first comparison as single-objective
        # search operators. Equal violations retain the historical incumbent.
        improved = self.bestObjs is None or compareSolutions(
            localBestObjs, localBestCons, self.bestObjs, self.bestCons, pop.conWgt
        ) < 0

        if improved:
            self.bestDecs = localBestDecs.copy()
            self.bestObjs = localBestObjs.copy()
            self.bestCons = None if localBestCons is None else localBestCons.copy()
            self.bestFeasible = localBestFeasible
            self.appearFEs = FEs
            self.appearIters = iters

        self.bestMetric = None
        return improved

    def observeMulti(self, pop, FEs, iters):
        """Archive evaluated real decisions before environmental selection."""
        if not len(pop):
            return
        feasible = pop.getParetoFront()
        changed = False
        if len(feasible):
            combined = feasible if self.archive is None else self.archive.merged(feasible)
            front = combined.getParetoFront()
            # One representative per objective vector, retaining the first observation.
            _, indices = np.unique(front.objs, axis=0, return_index=True)
            front = front[indices]
            changed = self.archive is None or not np.array_equal(front.objs, self.archive.objs)
            self.archive = front
            self.candidates = None
            self.minViolation = 0.0
        elif self.archive is None:
            candidates = pop.getInfeasibleCandidates(k=len(pop))
            if len(candidates):
                violation = calcConstraintViolation(candidates.cons, candidates.conWgt)
                minimum = float(violation.min())
                changed = self.minViolation is None or minimum < self.minViolation
                if changed:
                    self.minViolation = minimum
                    self.candidates = candidates[violation == minimum][:10]
        if changed:
            self.appearFEs = FEs
            self.appearIters = iters
            self._archiveImproved = True

    def _updateMulti(self, pop, FEs, iters):
        self.observeMulti(pop, FEs, iters)
        improved = self._archiveImproved
        self._archiveImproved = False
        if improved:
            # Evaluations may update the archive before this iteration commits.
            self.appearIters = iters
        front = pop[:0] if self.archive is None else self.archive
        self.bestDecs = front.decs.copy()
        self.bestObjs = front.objs.copy()
        self.bestCons = None if front.cons is None else front.cons.copy()
        self.bestFeasible = bool(len(front))
        if self.bestFeasible:
            refPoint = self._getHvRefPoint(front.objs)
            if improved or self.bestMetric is None:
                self.bestMetric = float(HV(front.objs, refPoint=refPoint, normalize=False,
                                           rng=np.random.default_rng(0)))
            direction = np.asarray(getattr(getattr(self.algorithm, "problem", None), "opt", 1))
            self.extra["hv_reference_point"] = refPoint * direction
            self.extra["hv_normalized"] = False
        else:
            self.bestMetric = None
        return improved

    def _updateHistory(self, pop, FEs, iters, improved):
        historyFreq = getattr(self.algorithm, "historyFreq", 1)
        if historyFreq is not None and (not self.history.iterToFEs or iters % historyFreq == 0):
            self._recordSnapshot(pop, FEs, iters)
        self.history.metrics.append(self.bestMetric)
        self.history.iterToFEs.append([iters, FEs])
        self.history.improvedHistory.append(bool(improved))

        if self.bestObjs is not None and self.bestObjs.shape[1] == 1:
            self.history.bestObjHistory.append(float(self.bestObjs[0, 0]))
        elif self.bestObjs is not None:
            self.history.numBestHistory.append(int(self.bestObjs.shape[0]))
            self.history.bestMetricHistory.append(self.bestMetric)

    def recordFinalSnapshot(self):
        if self.currentPop is not None and self.history.iterToFEs:
            iters, FEs = self.history.iterToFEs[-1]
            self._recordSnapshot(self.currentPop, FEs, iters)

    def _recordSnapshot(self, pop, FEs, iters):
        key = [iters, FEs]
        if self.history.snapshotIterToFEs and self.history.snapshotIterToFEs[-1] == key:
            self.history.populations.pop()
            self.history.bests.pop()
            self.history.snapshotIterToFEs.pop()
        self.history.snapshotIterToFEs.append(key)
        self.history.populations.append(
            {
                "decs": pop.decs.copy(),
                "constraint_weights": None if pop.conWgt is None else pop.conWgt.copy(),
                "objs": None if pop.objs is None else pop.objs.copy(),
                "cons": None if pop.cons is None else pop.cons.copy(),
            }
        )
        self.history.bests.append(
            {
                "bestDecs": None if self.bestDecs is None else self.bestDecs.copy(),
                "bestObjs": None if self.bestObjs is None else self.bestObjs.copy(),
                "bestCons": None if self.bestCons is None else self.bestCons.copy(),
                "candidateDecs": None if self.candidates is None else self.candidates.decs.copy(),
                "candidateObjs": None if self.candidates is None else self.candidates.objs.copy(),
                "candidateCons": None if self.candidates is None else self.candidates.cons.copy(),
                "minViolation": self.minViolation,
                "bestFeasible": self.bestFeasible,
            }
        )
    def buildResult(self, *, includeHistory=True):
        direction = np.asarray(getattr(self.algorithm.problem, "opt", 1))
        history = deepcopy(self.history) if includeHistory else OptHistory()
        for population in history.populations:
            if population["objs"] is not None:
                population["objs"] *= direction
        for best in history.bests:
            if best["bestObjs"] is not None:
                best["bestObjs"] *= direction
            if best.get("candidateObjs") is not None:
                best["candidateObjs"] *= direction
        if direction.size == 1:
            history.bestObjHistory = [value * float(direction.ravel()[0])
                                      for value in history.bestObjHistory]
        return OptResult(
            bestDecs=None if self.bestDecs is None else self.bestDecs.copy(),
            bestObjs=None if self.bestObjs is None else self.bestObjs * direction,
            bestCons=None if self.bestCons is None else self.bestCons.copy(),
            bestMetric=self.bestMetric,
            bestFeasible=self.bestFeasible,
            appearFEs=self.appearFEs,
            appearIters=self.appearIters,
            FEs=self.algorithm.FEs,
            iters=self.algorithm.iters,
            runtime=self.runtime,
            history=history,
            extra=deepcopy(self.extra),
            candidateDecs=None if self.candidates is None else self.candidates.decs.copy(),
            candidateObjs=None if self.candidates is None else self.candidates.objs * direction,
            candidateCons=None if self.candidates is None else self.candidates.cons.copy(),
            minViolation=self.minViolation,
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
        for name, key in (("candidateDecs", "candidate_decs"), ("candidateObjs", "candidate_objs"),
                          ("candidateCons", "candidate_cons"), ("minViolation", "min_violation")):
            value = getattr(result, name)
            if value is not None:
                payload[key] = np.asarray(value)
        if "hv_reference_point" in result.extra:
            payload["hv_reference_point"] = result.extra["hv_reference_point"]
        if self.history.bestObjHistory:
            payload["bestObjHistory"] = np.asarray(result.history.bestObjHistory, dtype=float)
        if self.history.numBestHistory:
            payload["numBestHistory"] = np.asarray(self.history.numBestHistory, dtype=np.int64)
        if self.history.bestMetricHistory:
            payload["bestMetricHistory"] = np.asarray(self.history.bestMetricHistory, dtype=float)
        weights = result.extra.get("constraint_weights")
        if weights is not None:
            payload["constraint_weights"] = np.asarray(weights).copy()
        return payload

    def reset(self):
        self.archive = None
        self.candidates = None
        self.minViolation = None
        self._archiveImproved = False
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
            reference = np.asarray(algRefPoint, dtype=float).reshape(-1)
            if reference.shape != (bestObjs.shape[1],) or not np.all(np.isfinite(reference)):
                raise ValueError("hvRefPoint must have one finite value per objective.")
            direction = np.asarray(getattr(getattr(self.algorithm, "problem", None), "opt", 1))
            self.hvRefPoint = (reference * direction).reshape(-1).copy()
            return self.hvRefPoint

        worst = np.max(np.asarray(bestObjs, dtype=float), axis=0)
        self.hvRefPoint = worst + np.where(worst == 0.0, 0.2, np.abs(worst) * 0.2)
        return self.hvRefPoint


Result = OptState
