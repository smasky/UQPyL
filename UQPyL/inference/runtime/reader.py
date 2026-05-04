import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np


def _from_json_array(text):
    if text is None:
        return None
    return np.asarray(json.loads(text))


class InfReader:
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
            try:
                run = conn.execute(
                    """
                    SELECT runId, method, problem, status, finalFEs, finalIters,
                           runtime, createdAt, finishedAt
                    FROM run
                    LIMIT 1
                    """
                ).fetchone()
            except sqlite3.OperationalError:
                run = None
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

    def getRunParams(self):
        rows = self.conn.execute("SELECT name, value FROM runParam ORDER BY name").fetchall()
        return {row["name"]: row["value"] for row in rows}

    def loadProblem(self):
        row = self.conn.execute("SELECT problemPayload FROM run LIMIT 1").fetchone()
        if row is None or row["problemPayload"] is None:
            raise ValueError("No problem payload found in sqlite database.")
        return pickle.loads(row["problemPayload"])

    def listSnapshots(self):
        rows = self.conn.execute(
            """
            SELECT snapshotId, iter, fe, elapsed, meanLogProb, bestObj,
                   feasibleRate, acceptanceRateMean
            FROM snapshot
            ORDER BY snapshotId
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def loadSnapshotMembers(self, snapshotId):
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
                "decs": _from_json_array(row["decs"]),
                "objs": _from_json_array(row["objs"]),
                "cons": _from_json_array(row["cons"]),
                "logProb": row["logProb"],
                "accepted": bool(row["accepted"]),
                "feasible": bool(row["feasible"]),
            }
            for row in rows
        ]

    def loadLastSnapshotMembers(self):
        row = self.conn.execute("SELECT snapshotId FROM snapshot ORDER BY snapshotId DESC LIMIT 1").fetchone()
        if row is None:
            raise ValueError("No snapshot found in sqlite database.")
        return self.loadSnapshotMembers(row["snapshotId"])

    def loadResult(self):
        row = self.conn.execute(
            "SELECT payload FROM artifact WHERE name = ? ORDER BY artifactId DESC LIMIT 1",
            ("result",),
        ).fetchone()
        if row is None:
            raise ValueError("No result artifact found in sqlite database.")
        return pickle.loads(row["payload"])
