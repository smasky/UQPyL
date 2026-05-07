import json
import pickle
import sqlite3

import numpy as np

from ...core.runtime import export_reader_summary, from_json_array
from ...core.runtime_reader import BaseReader


class InfReader(BaseReader):
    @classmethod
    def list_runs(cls, result_dir):
        return super().list_runs(
            result_dir,
            run_columns="runId, method, problem, status, finalFEs, finalIters, runtime, createdAt, finishedAt",
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
                "status": run["status"],
                "final_fes": run["finalFEs"],
                "final_iters": run["finalIters"],
            },
        )

    def load_problem(self):
        row = self.conn.execute("SELECT problemPayload FROM run LIMIT 1").fetchone()
        if row is None or row["problemPayload"] is None:
            raise ValueError("No problem payload found in sqlite database.")
        return pickle.loads(row["problemPayload"])

    def list_snapshots(self):
        rows = self.conn.execute(
            """
            SELECT snapshotId, iter, fe, elapsed, meanLogProb, bestObj,
                   feasibleRate, acceptanceRateMean
            FROM snapshot
            ORDER BY snapshotId
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def load_snapshot_members(self, snapshotId):
        rows = self.conn.execute(
            """
            SELECT chain, decs, objs, cons, logProb, accepted, feasible
            FROM snapshotMember
            WHERE snapshotId = ?
            ORDER BY chain
            """,
            (snapshotId,),
        ).fetchall()
        return [
            {
                "chain": row["chain"],
                "decs": from_json_array(row["decs"]),
                "objs": from_json_array(row["objs"]),
                "cons": from_json_array(row["cons"]),
                "logProb": row["logProb"],
                "accepted": bool(row["accepted"]),
                "feasible": bool(row["feasible"]),
            }
            for row in rows
        ]

    def load_last_snapshot_members(self):
        row = self.conn.execute("SELECT snapshotId FROM snapshot ORDER BY snapshotId DESC LIMIT 1").fetchone()
        if row is None:
            raise ValueError("No snapshot found in sqlite database.")
        return self.load_snapshot_members(row["snapshotId"])

    def load_result(self):
        row = self.conn.execute(
            "SELECT payload FROM artifact WHERE name = ? ORDER BY artifactId DESC LIMIT 1",
            ("result",),
        ).fetchone()
        if row is None:
            raise ValueError("No result artifact found in sqlite database.")
        return pickle.loads(row["payload"])
