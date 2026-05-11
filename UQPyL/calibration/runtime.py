from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Any

import numpy as np

from ..core.runtime import export_runtime_meta
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
    nTime: int
    nSeries: int
    nObs: int
    settings: dict[str, Any]
    runtime: float
    createdAt: str
    obs: np.ndarray
    mask: np.ndarray
    simLabels: list[str]
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

    def summary(self) -> dict[str, Any]:
        bestScore = self.diagnostics.get("scores")
        bestScoreValue = None
        if isinstance(bestScore, np.ndarray) and bestScore.size > 0:
            bestIdx = int(self.extra.get("bestIdx", 0))
            bestScoreValue = float(bestScore[bestIdx])
        elif np.isscalar(bestScore):
            bestScoreValue = float(bestScore)

        return export_runtime_meta(
            run_id=self.runId,
            method=self.method,
            problem_name=self.problemName,
            n_input=self.nInput,
            n_output=self.nSeries,
            n_con=0,
            runtime=self.runtime,
            created_at=self.createdAt,
            extra={
                "n_time": self.nTime,
                "n_series": self.nSeries,
                "n_obs": self.nObs,
                "metric": self.settings.get("metric"),
                "best_score": bestScoreValue,
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
        self.history.reset()

    def buildResult(self):
        problem = self.calibration.problem
        return CalResult(
            runId=getattr(self.calibration, "runId", None),
            method=self.calibration.name,
            problemName=problem.name,
            nInput=problem.nInput,
            nTime=problem.obs.shape[0],
            nSeries=problem.obs.shape[1],
            nObs=problem.obs.size,
            settings=self.calibration.params.asDict(),
            runtime=float(self.runtime),
            createdAt=self.createdAt,
            obs=problem.obs.copy(),
            mask=problem.mask.copy() if problem.mask is not None else np.zeros(problem.obs.shape, dtype=bool),
            simLabels=list(problem.simLabels),
            bestDecs=None if self.bestDecs is None else self.bestDecs.copy(),
            bestSim=None if self.bestSim is None else self.bestSim.copy(),
            posteriorDecs=None if self.posteriorDecs is None else self.posteriorDecs.copy(),
            posteriorSims=None if self.posteriorSims is None else self.posteriorSims.copy(),
            behavioralDecs=None if self.behavioralDecs is None else self.behavioralDecs.copy(),
            behavioralSims=None if self.behavioralSims is None else self.behavioralSims.copy(),
            eliteDecs=None if self.eliteDecs is None else self.eliteDecs.copy(),
            eliteSims=None if self.eliteSims is None else self.eliteSims.copy(),
            diagnostics=dict(self.diagnostics),
            history=self.history,
            extra=dict(self.extra),
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
    resultDir = os.path.join(workDir, "Result")
    os.makedirs(resultDir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileName = f"{result.method.lower()}_{timestamp}.log"
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
    def _makeRunId(self, methodName, problemName):
        _, runId = self._db_path(methodName, problemName)
        return runId

    def _insert_run(self, conn, runId, obj, now):
        problem = obj.problem
        conn.execute(
            """
            INSERT INTO run (
                runId, method, problem, nInput, nTime, nSeries, nObs,
                status, runtime, createdAt, finishedAt, problemPayload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                runId,
                obj.name,
                problem.name,
                problem.nInput,
                problem.obs.shape[0],
                problem.obs.shape[1],
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
        self.finalize_run(session, status="finished", runtime=result.runtime)

        artifacts = {
            "result": result,
            "obs": result.obs,
            "mask": result.mask,
            "simLabels": result.simLabels,
            "bestDecs": result.bestDecs,
            "bestSim": result.bestSim,
            "posteriorDecs": result.posteriorDecs,
            "posteriorSims": result.posteriorSims,
            "behavioralDecs": result.behavioralDecs,
            "behavioralSims": result.behavioralSims,
            "eliteDecs": result.eliteDecs,
            "eliteSims": result.eliteSims,
            "diagnostics": result.diagnostics,
            "history": result.history,
            "settings": result.settings,
            "extra": result.extra,
        }
        for name, payload in artifacts.items():
            conn.execute(
                "INSERT INTO artifact (runId, name, payload) VALUES (?, ?, ?)",
                (
                    runId,
                    name,
                    None if payload is None else self._problem_blob(payload),
                ),
            )

        conn.commit()

    def _create_schema(self, conn):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run (
                runId TEXT PRIMARY KEY,
                method TEXT NOT NULL,
                problem TEXT NOT NULL,
                nInput INTEGER NOT NULL,
                nTime INTEGER NOT NULL,
                nSeries INTEGER NOT NULL,
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
