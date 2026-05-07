import json
import pickle
import sqlite3

import numpy as np

from ...core.runtime import export_reader_summary, from_json_array
from ...core.runtime_reader import BaseReader
from .result import AnaMetric, AnaResult


class AnaReader(BaseReader):
    @classmethod
    def list_runs(cls, result_dir):
        return super().list_runs(
            result_dir,
            run_columns="runId, method, problem, target, status, runtime, createdAt, finishedAt",
        )

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

    def get_run(self):
        row = self.conn.execute("SELECT * FROM run LIMIT 1").fetchone()
        return dict(row) if row is not None else None

    def get_run_summary(self):
        run = self.get_run()
        if run is None:
            raise ValueError("No run record found in sqlite database.")

        metrics = self.get_metrics()
        artifacts = self.get_artifacts()
        return export_reader_summary(
            run_id=run["runId"],
            method=run["method"],
            problem_name=run["problem"],
            n_input=run["nInput"],
            n_output=run["nOutput"],
            n_con=run["nCon"],
            runtime=0.0 if run["runtime"] is None else float(run["runtime"]),
            created_at=run["createdAt"],
            finished_at=run["finishedAt"],
            extra={
                "target": run["target"],
                "metric_names": [metric.name for metric in metrics],
                "artifact_names": sorted(artifacts.keys()),
            },
        )

    def get_run_params(self):
        """
        Return raw parameter strings stored in sqlite.

        Values come from `repr(value)` during persistence and are not
        converted back to original Python types here.
        """
        rows = self.conn.execute("SELECT name, value FROM runParam ORDER BY name").fetchall()
        return {row["name"]: row["value"] for row in rows}

    def get_metrics(self):
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
                    values=from_json_array(row["valueJson"]),
                    rowLabels=json.loads(row["rowLabels"]),
                    colLabels=json.loads(row["colLabels"]),
                    colDim=row["colDim"],
                )
            )
        return metrics

    def get_metric(self, name):
        return self.load_result().getMetric(name)

    def get_artifacts(self):
        rows = self.conn.execute(
            """
            SELECT name, payload
            FROM artifact
            ORDER BY artifactId
            """
        ).fetchall()
        return {row["name"]: pickle.loads(row["payload"]) for row in rows}

    def load_problem(self):
        row = self.conn.execute("SELECT problemPayload FROM run LIMIT 1").fetchone()
        if row is None or row["problemPayload"] is None:
            raise ValueError("No problem payload found in sqlite database.")
        return pickle.loads(row["problemPayload"])

    def load_result(self):
        run = self.get_run()
        if run is None:
            raise ValueError("No run record found in sqlite database.")

        artifacts = self.get_artifacts()
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
            metrics=self.get_metrics(),
            X=artifacts.get("X"),
            Y=artifacts.get("Y"),
            runtime=0.0 if run["runtime"] is None else float(run["runtime"]),
            createdAt=run["createdAt"],
            extra=extra,
        )
