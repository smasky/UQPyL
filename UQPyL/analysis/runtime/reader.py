import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np

from .result import AnaMetric, AnaResult


def _from_json_array(text):
    if text is None:
        return None
    return np.asarray(json.loads(text))


class AnaReader:
    @staticmethod
    def listRuns(resultDir):
        resultPath = Path(resultDir)
        if resultPath.is_file():
            resultPath = resultPath.parent
        if resultPath.name.lower() != "result" and (resultPath / "Result").exists():
            resultPath = resultPath / "Result"

        rows = []
        for dbPath in sorted(resultPath.glob("*.sqlite3")):
            conn = sqlite3.connect(dbPath)
            conn.row_factory = sqlite3.Row
            run = conn.execute(
                """
                SELECT runId, method, problem, target, status, runtime, createdAt, finishedAt
                FROM run
                LIMIT 1
                """
            ).fetchone()
            conn.close()
            if run is None:
                continue
            item = dict(run)
            item["dbPath"] = str(dbPath)
            item["fileName"] = dbPath.name
            rows.append(item)
        return rows

    def __init__(self, dbPath):
        self.dbPath = str(dbPath)
        self.conn = sqlite3.connect(self.dbPath)
        self.conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.conn.close()

    def getRun(self):
        row = self.conn.execute("SELECT * FROM run LIMIT 1").fetchone()
        return dict(row) if row is not None else None

    def getRunSummary(self):
        run = self.getRun()
        if run is None:
            raise ValueError("No run record found in sqlite database.")

        metrics = self.getMetrics()
        artifacts = self.getArtifacts()
        return {
            "runId": run["runId"],
            "method": run["method"],
            "problemName": run["problem"],
            "target": run["target"],
            "nInput": run["nInput"],
            "nOutput": run["nOutput"],
            "nCon": run["nCon"],
            "runtime": 0.0 if run["runtime"] is None else float(run["runtime"]),
            "createdAt": run["createdAt"],
            "finishedAt": run["finishedAt"],
            "metricNames": [metric.name for metric in metrics],
            "artifactNames": sorted(artifacts.keys()),
        }

    def getRunParams(self):
        """
        Return raw parameter strings stored in sqlite.

        Values come from `repr(value)` during persistence and are not
        converted back to original Python types here.
        """
        rows = self.conn.execute("SELECT name, value FROM runParam ORDER BY name").fetchall()
        return {row["name"]: row["value"] for row in rows}

    def getMetrics(self):
        rows = self.conn.execute(
            """
            SELECT name, rowLabels, colLabels, valueJson, colDim
            FROM metric
            ORDER BY metricId
            """
        ).fetchall()
        metrics = []
        for row in rows:
            metrics.append(
                AnaMetric(
                    name=row["name"],
                    values=_from_json_array(row["valueJson"]),
                    rowLabels=json.loads(row["rowLabels"]),
                    colLabels=json.loads(row["colLabels"]),
                    colDim=row["colDim"],
                )
            )
        return metrics

    def getMetric(self, name):
        return self.loadResult().getMetric(name)

    def getArtifacts(self):
        rows = self.conn.execute(
            """
            SELECT name, payload
            FROM artifact
            ORDER BY artifactId
            """
        ).fetchall()
        return {row["name"]: pickle.loads(row["payload"]) for row in rows}

    def loadProblem(self):
        row = self.conn.execute("SELECT problemPayload FROM run LIMIT 1").fetchone()
        if row is None or row["problemPayload"] is None:
            raise ValueError("No problem payload found in sqlite database.")
        return pickle.loads(row["problemPayload"])

    def loadResult(self):
        run = self.getRun()
        if run is None:
            raise ValueError("No run record found in sqlite database.")

        artifacts = self.getArtifacts()
        settings = artifacts.get("settings", {})
        meta = artifacts.get("meta", {})
        extra = artifacts.get("extra", {})

        return AnaResult(
            runId=run["runId"],
            method=run["method"],
            problemName=run["problem"],
            nInput=run["nInput"],
            nOutput=run["nOutput"],
            nCon=run["nCon"],
            target=run["target"],
            settings=settings,
            meta=meta,
            metrics=self.getMetrics(),
            X=artifacts.get("X"),
            Y=artifacts.get("Y"),
            runtime=0.0 if run["runtime"] is None else float(run["runtime"]),
            createdAt=run["createdAt"],
            extra=extra,
        )
