from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class AnaMetric:
    name: str
    values: np.ndarray
    rowLabels: list[str]
    colLabels: list[str]
    colDim: str

    def toDict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": self.values.copy(),
            "rowLabels": list(self.rowLabels),
            "colLabels": list(self.colLabels),
            "colDim": self.colDim,
        }


@dataclass
class AnaResult:
    runId: str | None
    method: str
    problemName: str
    nInput: int
    nOutput: int
    nCon: int
    target: str
    settings: dict[str, Any]
    meta: dict[str, Any]
    metrics: list[AnaMetric]
    X: np.ndarray | None
    Y: np.ndarray | None
    runtime: float
    createdAt: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def metricNames(self) -> list[str]:
        return [metric.name for metric in self.metrics]

    @property
    def metricMap(self) -> dict[str, AnaMetric]:
        return {metric.name: metric for metric in self.metrics}

    def getMetric(self, name: str) -> AnaMetric:
        metric = self.metricMap.get(name)
        if metric is None:
            raise KeyError(f"Metric '{name}' not found in analysis result.")
        return metric

    def __getitem__(self, name: str) -> AnaMetric:
        return self.getMetric(name)

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "runId": self.runId,
            "problemName": self.problemName,
            "target": self.target,
            "nInput": self.nInput,
            "nOutput": self.nOutput,
            "nCon": self.nCon,
            "metricNames": self.metricNames,
            "runtime": self.runtime,
            "createdAt": self.createdAt,
        }

    def toDict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "runId": self.runId,
            "problemName": self.problemName,
            "nInput": self.nInput,
            "nOutput": self.nOutput,
            "nCon": self.nCon,
            "target": self.target,
            "settings": dict(self.settings),
            "meta": dict(self.meta),
            "metrics": [metric.toDict() for metric in self.metrics],
            "X": None if self.X is None else self.X.copy(),
            "Y": None if self.Y is None else self.Y.copy(),
            "runtime": self.runtime,
            "createdAt": self.createdAt,
            "extra": dict(self.extra),
        }


class AnaState:
    def __init__(self, analysis):
        self.analysis = analysis
        self.reset()

    def reset(self):
        self.X = None
        self.Y = None
        self.target = "objs"
        self.meta = {}
        self.metrics: list[AnaMetric] = []
        self.runtime = 0.0
        self.createdAt = datetime.now().isoformat(timespec="seconds")
        self.extra = {}
        self.verbose = {}

    def record(self, X, Y, metrics, target="objs", meta=None):
        self.X = None if X is None else np.asarray(X).copy()
        self.Y = None if Y is None else np.asarray(Y).copy()
        self.target = target
        self.meta = {} if meta is None else dict(meta)
        self.metrics = [
            AnaMetric(
                name=name,
                values=np.asarray(values).copy(),
                rowLabels=list(rowLabels),
                colLabels=list(colLabels),
                colDim=colDim,
            )
            for name, values, rowLabels, colLabels, colDim in metrics
        ]

    def buildResult(self) -> AnaResult:
        problem = self.analysis.problem
        return AnaResult(
            runId=getattr(self.analysis, "runId", None),
            method=self.analysis.name,
            problemName=problem.name,
            nInput=problem.nInput,
            nOutput=problem.nOutput,
            nCon=problem.nCons,
            target=self.target,
            settings=self.analysis.setting.asDict(),
            meta=dict(self.meta),
            metrics=list(self.metrics),
            X=None if self.X is None else self.X.copy(),
            Y=None if self.Y is None else self.Y.copy(),
            runtime=float(self.runtime),
            createdAt=self.createdAt,
            extra=self.extra.copy(),
        )
