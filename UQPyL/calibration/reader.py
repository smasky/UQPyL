import pickle
import sqlite3

from ..core.runtime import export_reader_summary
from ..core.runtime_reader import BaseReader


class CalReader(BaseReader):
    domain = 'calibration'
    @classmethod
    def list_runs(cls, result_dir):
        return super().list_runs(
            result_dir,
            run_columns="runId, method, problem, status, runtime, createdAt, finishedAt",
        )







    def get_run_summary(self):
        run = self.get_run()
        if run is None:
            raise ValueError("No run record found in sqlite database.")

        row = self.conn.execute("SELECT payload FROM artifact WHERE name='summary' ORDER BY artifactId DESC LIMIT 1").fetchone()
        summary = {} if row is None or row[0] is None else pickle.loads(row[0])
        metric = summary.get('metric')
        bestScore = summary.get('best_score')
        artifactNames = [row[0] for row in self.conn.execute('SELECT name FROM artifact ORDER BY name')]

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
                "artifact_names": artifactNames,
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
