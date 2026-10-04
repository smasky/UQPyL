import json
import pickle
import sqlite3

import numpy as np

from ...core.runtime import export_reader_summary, from_json_array
from ...core.runtime_reader import BaseReader


class InfReader(BaseReader):
    domain = "inference"

    @classmethod
    def list_runs(cls, result_dir):
        return super().list_runs(
            result_dir,
            run_columns="runId, method, problem, status, finalFEs, finalIters, runtime, createdAt, finishedAt",
        )

    def get_run_summary(self):
        run = self.get_run()
        if run is None:
            raise ValueError("No run record found in sqlite database.")
        snapshot = self.conn.execute("SELECT statePayload FROM snapshot ORDER BY snapshotId DESC LIMIT 1").fetchone()
        stopReason = None if snapshot is None else json.loads(snapshot[0]).get("stop_reason")
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
                "stop_reason": stopReason,
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

    def load_partial_result(self):
        """Read saved chain endpoints, including failed runs, without inventing draws.

        Returns:
            dict: Run summary and sparse snapshots in real decision coordinates
            and original objective direction. This is not a resumable checkpoint
            or a complete chain, even when the run has finished.
        """
        snapshots = []
        for row in self.list_snapshots():
            members = self.load_snapshot_members(row["snapshotId"])
            snapshots.append(
                {
                    "snapshot_id": row["snapshotId"],
                    "iter": row["iter"],
                    "fes": row["fe"],
                    "runtime": row["elapsed"],
                    "members": [
                        {
                            **{key: value for key, value in member.items() if key != "logProb"},
                            "log_prob": member["logProb"],
                        }
                        for member in members
                    ],
                }
            )
        return {
            **self.get_run_summary(),
            "complete": False,
            "resumable": False,
            "sample_scope": "saved_chain_endpoints",
            "snapshot_count": len(snapshots),
            "last_saved_iter": None if not snapshots else snapshots[-1]["iter"],
            "last_saved_fes": None if not snapshots else snapshots[-1]["fes"],
            "snapshots": snapshots,
        }
