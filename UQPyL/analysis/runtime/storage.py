from __future__ import annotations

import json
import os
import pickle
import re
import sqlite3
import uuid
from datetime import datetime

import numpy as np

from .result import AnaResult


def _json_dumps(value):
    return json.dumps(value, ensure_ascii=True)


def _array_to_json(value):
    if value is None:
        return None
    return _json_dumps(np.asarray(value).tolist())


def _slugify_name(name):
    text = str(name).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "problem"


class SqliteStorage:
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.resultDir = os.path.join(rootDir, "Result")
        os.makedirs(self.resultDir, exist_ok=True)

    def _makeRunId(self, methodName, problemName):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = uuid.uuid4().hex[:4]
        problemSlug = _slugify_name(problemName)
        return f"{methodName.lower()}_{problemSlug}_{timestamp}_{suffix}"

    def _dbPath(self, methodName, problemName):
        runId = self._makeRunId(methodName, problemName)
        return os.path.join(self.resultDir, f"{runId}.sqlite3"), runId

    def createRun(self, obj):
        dbPath, runId = self._dbPath(obj.name, obj.problem.name)
        conn = sqlite3.connect(dbPath)
        self._createSchema(conn)
        now = datetime.now().isoformat(timespec="seconds")
        problem = obj.problem

        conn.execute(
            """
            INSERT INTO run (
                runId, method, problem, target, nInput, nOutput, nCon,
                status, runtime, createdAt, finishedAt, problemPayload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                runId,
                obj.name,
                problem.name,
                None,
                problem.nInput,
                problem.nOutput,
                problem.nCons,
                "running",
                0.0,
                now,
                None,
                sqlite3.Binary(pickle.dumps(problem, protocol=pickle.HIGHEST_PROTOCOL)),
            ),
        )

        for name, value in obj.setting.items():
            conn.execute(
                "INSERT INTO runParam (runId, name, value) VALUES (?, ?, ?)",
                (runId, name, repr(value)),
            )

        conn.commit()
        return {"conn": conn, "dbPath": dbPath, "runId": runId}

    def saveResult(self, storageCtx, result: AnaResult):
        conn = storageCtx["conn"]
        runId = storageCtx["runId"]
        finishedAt = datetime.now().isoformat(timespec="seconds")

        conn.execute(
            """
            UPDATE run
            SET target=?, status=?, runtime=?, finishedAt=?
            WHERE runId=?
            """,
            (result.target, "finished", result.runtime, finishedAt, runId),
        )

        for metric in result.metrics:
            conn.execute(
                """
                INSERT INTO metric (runId, name, rowLabels, colLabels, valueJson, colDim)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    runId,
                    metric.name,
                    _json_dumps(metric.rowLabels),
                    _json_dumps(metric.colLabels),
                    _array_to_json(metric.values),
                    metric.colDim,
                ),
            )

        artifacts = {
            "X": result.X,
            "Y": result.Y,
            "settings": result.settings,
            "meta": result.meta,
            "extra": result.extra,
        }
        for name, payload in artifacts.items():
            conn.execute(
                "INSERT INTO artifact (runId, name, payload) VALUES (?, ?, ?)",
                (
                    runId,
                    name,
                    sqlite3.Binary(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)),
                ),
            )

        conn.commit()

    def close(self, storageCtx):
        storageCtx["conn"].close()

    def _createSchema(self, conn):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run (
                runId TEXT PRIMARY KEY,
                method TEXT NOT NULL,
                problem TEXT NOT NULL,
                target TEXT,
                nInput INTEGER NOT NULL,
                nOutput INTEGER NOT NULL,
                nCon INTEGER NOT NULL,
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

            CREATE TABLE IF NOT EXISTS metric (
                metricId INTEGER PRIMARY KEY AUTOINCREMENT,
                runId TEXT NOT NULL,
                name TEXT NOT NULL,
                rowLabels TEXT,
                colLabels TEXT,
                valueJson TEXT,
                colDim TEXT NOT NULL,
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
