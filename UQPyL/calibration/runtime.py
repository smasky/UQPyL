from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Any

import numpy as np

from ..core.runtime import ensure_result_dir, export_runtime_meta
from ..core.runtime_storage import BaseSqliteStorage


@dataclass
class CalHistory:
    metricsHistory: list[dict[str, Any]] = field(default_factory=list)

    def reset(self):
        self.metricsHistory.clear()


@dataclass
class CalResult:
    runId: str | None
    method: str
    problemName: str
    nInput: int
    nObs: int
    settings: dict[str, Any]
    runtime: float
    createdAt: str
    obs: np.ndarray
    mask: np.ndarray
    bestDecs: np.ndarray | None = None
    bestSim: np.ndarray | None = None
    posteriorDecs: np.ndarray | None = None
    posteriorSims: np.ndarray | None = None
    behavioralDecs: np.ndarray | None = None
    behavioralSims: np.ndarray | None = None
    eliteDecs: np.ndarray | None = None
    eliteSims: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    history: CalHistory = field(default_factory=CalHistory)
    extra: dict[str, Any] = field(default_factory=dict)
    samples: np.ndarray | None = None
    simulations: np.ndarray | None = None
    scores: np.ndarray | None = None
    sample_kind: str | None = None
    weights: np.ndarray | None = None
    intervals: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: dict[str, Any] | None = None
    best_score: float | None = None
    best_index: int | None = None

    def summary(self) -> dict[str, Any]:
        return export_runtime_meta(
            run_id=self.runId,
            method=self.method,
            problem_name=self.problemName,
            n_input=self.nInput,
            n_output=self.nObs,
            n_con=0,
            runtime=self.runtime,
            created_at=self.createdAt,
            extra={
                "n_obs": self.nObs,
                "metric": self.settings.get("metric"),
                "best_score": self.best_score,
                "best_index": self.best_index,
                "sample_kind": self.sample_kind,
                "n_samples": 0 if self.samples is None else len(self.samples),
                "has_weights": self.weights is not None,
                "interval_count": len(self.intervals),
                "best_x": None if self.bestDecs is None else self.bestDecs.reshape(-1).copy(),
                "iters": len(self.history.metricsHistory),
            },
        )


class CalState:
    def __init__(self, calibration):
        self.calibration = calibration
        self.history = CalHistory()
        self.reset()

    def reset(self):
        self.runtime = 0.0
        self.createdAt = datetime.now().isoformat(timespec="seconds")
        self.bestDecs = None
        self.bestSim = None
        self.posteriorDecs = None
        self.posteriorSims = None
        self.behavioralDecs = None
        self.behavioralSims = None
        self.eliteDecs = None
        self.eliteSims = None
        self.diagnostics = {}
        self.extra = {}
        self.sampleKind = None
        self.history.reset()

    def buildResult(self):
        """Build a snapshot independent of mutable runtime state and settings."""
        problem = self.calibration.problem
        unified = self._unifiedOutputs(problem)
        return CalResult(
            runId=getattr(self.calibration, "runId", None),
            method=self.calibration.name,
            problemName=problem.name,
            nInput=problem.nInput,
            nObs=problem.obs.size,
            settings=deepcopy(self.calibration.params.asDict()),
            runtime=float(self.runtime),
            createdAt=self.createdAt,
            obs=problem.obs.copy(),
            mask=problem.mask.copy() if problem.mask is not None else np.zeros(problem.obs.shape, dtype=bool),
            bestDecs=None if self.bestDecs is None else self.bestDecs.copy(),
            bestSim=None if self.bestSim is None else self.bestSim.copy(),
            posteriorDecs=None if self.posteriorDecs is None else self.posteriorDecs.copy(),
            posteriorSims=None if self.posteriorSims is None else self.posteriorSims.copy(),
            behavioralDecs=None if self.behavioralDecs is None else self.behavioralDecs.copy(),
            behavioralSims=None if self.behavioralSims is None else self.behavioralSims.copy(),
            eliteDecs=None if self.eliteDecs is None else self.eliteDecs.copy(),
            eliteSims=None if self.eliteSims is None else self.eliteSims.copy(),
            diagnostics=deepcopy(self.diagnostics),
            history=deepcopy(self.history),
            extra=deepcopy(self.extra),
            **unified,
        )

    def _unifiedOutputs(self, problem):
        """Align primary rows and label interval provenance without simulation."""
        behavioral = self.sampleKind == "behavioral"
        samples = self.behavioralDecs if behavioral else self.posteriorDecs
        simulations = self.behavioralSims if behavioral else self.posteriorSims
        scores = self.diagnostics.get("behavioralScores" if behavioral else "scores")
        weights = self.diagnostics.get("behavioralWeights") if behavioral else None
        allScores = self.diagnostics.get("scores")
        bestScore = None
        bestIndex = None
        if allScores is not None and np.size(allScores):
            originalIndex = int(self.extra.get("bestIdx", 0))
            bestScore = float(np.asarray(allScores).reshape(-1)[originalIndex])
            if behavioral:
                selected = np.flatnonzero(self.diagnostics["behavioralMask"])
                matches = np.flatnonzero(selected == originalIndex)
                bestIndex = int(matches[0]) if matches.size else None
            elif samples is not None:
                bestIndex = originalIndex
        mask = np.zeros(problem.obs.shape, dtype=bool) if problem.mask is None else problem.mask
        validIndices = np.flatnonzero(~mask.reshape(-1))
        intervals = []

        def addInterval(kind, space, lower, upper, probability, source):
            if lower is None or upper is None:
                return
            intervals.append(
                {
                    "kind": kind,
                    "space": space,
                    "lower": np.asarray(lower).copy(),
                    "upper": np.asarray(upper).copy(),
                    "probability": probability,
                    "sample_source": source,
                    "indices": validIndices.copy() if space == "simulation" else np.arange(problem.nInput),
                }
            )

        addInterval(
            "weighted_empirical" if behavioral else "sampling_envelope",
            "simulation",
            self.diagnostics.get("ppuLower"),
            self.diagnostics.get("ppuUpper"),
            self.diagnostics.get("interval", 0.95),
            "samples",
        )
        uncertainty = deepcopy(self.extra.get("uncertainty"))
        if uncertainty is not None:
            for space in ("parameter", "simulation"):
                addInterval(
                    "prior_importance_weighting",
                    space,
                    uncertainty.get(f"{space}_lower"),
                    uncertainty.get(f"{space}_upper"),
                    uncertainty["interval"],
                    "uncertainty.samples",
                )
        return dict(
            samples=None if samples is None else samples.copy(),
            simulations=None if simulations is None else simulations.copy(),
            scores=None if scores is None else np.asarray(scores).copy(),
            sample_kind=self.sampleKind,
            weights=None if weights is None else weights.copy(),
            intervals=intervals,
            uncertainty=uncertainty,
            best_score=bestScore,
            best_index=bestIndex,
        )


def format_summary(result: CalResult) -> str:
    summary = result.summary()
    lines = [f"{result.method} finished"]
    ordered = [
        ("problem", summary["problem_name"]),
        ("metric", summary["metric"]),
        ("bestScore", _fmt(summary["best_score"])),
        ("bestX", _fmt_vector(summary["best_x"])),
        ("iters", summary["iters"]),
        ("runtime", f"{summary['runtime']:.3f}s"),
    ]

    if "pfactor" in result.diagnostics:
        ordered.append(("pfactor", _fmt(result.diagnostics["pfactor"])))
    if "rfactor" in result.diagnostics:
        ordered.append(("rfactor", _fmt(result.diagnostics["rfactor"])))
    if "posteriorMean" in result.diagnostics:
        ordered.append(("posteriorMean", _fmt_vector(result.diagnostics["posteriorMean"])))

    width = max(len(key) for key, _ in ordered)
    lines.extend(f"  {key.ljust(width)} : {value}" for key, value in ordered)
    return "\n".join(lines)


def save_log(result: CalResult, workDir: str):
    resultDir = ensure_result_dir(workDir)
    os.makedirs(resultDir, exist_ok=True)
    fileName = f"{result.runId}.log"
    filePath = os.path.join(resultDir, fileName)
    with open(filePath, "w", encoding="utf-8") as f:
        f.write(format_summary(result) + "\n")
        if result.history.metricsHistory:
            f.write("\n[history]\n")
            for item in result.history.metricsHistory:
                f.write(str(item) + "\n")
    return filePath


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return f"{value:.4e}"
    return str(value)


def _fmt_vector(value):
    if value is None:
        return "-"
    arr = np.asarray(value).reshape(-1)
    return "[" + ", ".join(_fmt(float(v)) for v in arr) + "]"


class SqliteStorage(BaseSqliteStorage):
    domain = "calibration"

    def _makeRunId(self, methodName, problemName):
        _, runId = self._db_path(methodName, problemName)
        return runId

    def _insert_run(self, conn, runId, obj, now):
        problem = obj.problem
        conn.execute(
            """
            INSERT INTO run (
                runId, method, problem, nInput, nObs,
                status, runtime, createdAt, finishedAt, problemPayload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                runId,
                obj.name,
                problem.name,
                problem.nInput,
                problem.obs.size,
                "running",
                0.0,
                now,
                None,
                self._problem_blob(problem),
            ),
        )

    def saveResult(self, session, result: CalResult):
        conn = session.conn
        runId = session.run_id

        artifacts = {"result": result, "summary": result.summary()}
        for name, payload in artifacts.items():
            conn.execute(
                "INSERT INTO artifact (runId, name, payload) VALUES (?, ?, ?)",
                (
                    runId,
                    name,
                    None if payload is None else self._problem_blob(payload),
                ),
            )

        self.finalize_run(session, status="finished", runtime=result.runtime)

    def _create_schema(self, conn):
        conn.executescript(
            """
            PRAGMA user_version = 1;

            CREATE TABLE IF NOT EXISTS run (
                runId TEXT PRIMARY KEY,
                method TEXT NOT NULL,
                problem TEXT NOT NULL,
                nInput INTEGER NOT NULL,
                nObs INTEGER NOT NULL,
                status TEXT NOT NULL,
                runtime REAL,
                createdAt TEXT NOT NULL,
                finishedAt TEXT,
                problemPayload BLOB
            );

            CREATE TABLE IF NOT EXISTS runParam (
                runId TEXT NOT NULL,
                name TEXT NOT NULL,
                value TEXT,
                FOREIGN KEY(runId) REFERENCES run(runId)
            );

            CREATE TABLE IF NOT EXISTS artifact (
                artifactId INTEGER PRIMARY KEY AUTOINCREMENT,
                runId TEXT NOT NULL,
                name TEXT NOT NULL,
                payload BLOB,
                FOREIGN KEY(runId) REFERENCES run(runId)
            );
            """
        )
