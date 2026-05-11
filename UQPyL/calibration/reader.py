import pickle
import sqlite3

from ..core.runtime import export_reader_summary
from ..core.runtime_reader import BaseReader


class CalReader(BaseReader):
    @classmethod
    def list_runs(cls, result_dir):
        return super().list_runs(
            result_dir,
            run_columns="runId, method, problem, status, runtime, createdAt, finishedAt",
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

    def get_run_params(self):
        rows = self.conn.execute("SELECT name, value FROM runParam ORDER BY name").fetchall()
        return {row["name"]: row["value"] for row in rows}

    def get_run_summary(self):
        run = self.get_run()
        if run is None:
            raise ValueError("No run record found in sqlite database.")

        artifacts = self.get_artifacts()
        result = artifacts.get("result")
        metric = None if result is None else result.settings.get("metric")
        bestScore = None if result is None else result.summary().get("best_score")

        return export_reader_summary(
            run_id=run["runId"],
            method=run["method"],
            problem_name=run["problem"],
            n_input=run["nInput"],
            n_output=run["nSeries"],
            n_con=0,
            runtime=0.0 if run["runtime"] is None else float(run["runtime"]),
            created_at=run["createdAt"],
            finished_at=run["finishedAt"],
            extra={
                "status": run["status"],
                "n_time": run["nTime"],
                "n_series": run["nSeries"],
                "n_obs": run["nObs"],
                "metric": metric,
                "best_score": bestScore,
                "artifact_names": sorted(artifacts.keys()),
            },
        )

    def get_artifacts(self):
        rows = self.conn.execute(
            """
            SELECT name, payload
            FROM artifact
            ORDER BY artifactId
            """
        ).fetchall()
        return {
            row["name"]: None if row["payload"] is None else pickle.loads(row["payload"])
            for row in rows
        }

    def load_problem(self):
        row = self.conn.execute("SELECT problemPayload FROM run LIMIT 1").fetchone()
        if row is None or row["problemPayload"] is None:
            raise ValueError("No problem payload found in sqlite database.")
        return pickle.loads(row["problemPayload"])

    def load_result(self):
        row = self.conn.execute(
            "SELECT payload FROM artifact WHERE name = ? ORDER BY artifactId DESC LIMIT 1",
            ("result",),
        ).fetchone()
        if row is None or row["payload"] is None:
            raise ValueError("No result artifact found in sqlite database.")
        return pickle.loads(row["payload"])
