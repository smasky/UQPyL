from __future__ import annotations

import json

import numpy as np

from ...core.runtime import array_to_json
from ...core.runtime_storage import BaseSqliteStorage
from .result import AnaResult


def _json_dumps(value):
    return json.dumps(value, ensure_ascii=True)


class SqliteStorage(BaseSqliteStorage):
    def _makeRunId(self, methodName, problemName):
        _, runId = self._db_path(methodName, problemName)
        return runId

    def _insert_run(self, conn, runId, obj, now):
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
                self._problem_blob(problem),
            ),
        )

    def saveResult(self, session, result: AnaResult):
        conn = session.conn
        runId = session.run_id
        self.finalize_run(session, status="finished", runtime=result.runtime)
        conn.execute("UPDATE run SET target=? WHERE runId=?", (result.target, runId))

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
                    array_to_json(metric.values),
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
                    self._problem_blob(payload),
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
