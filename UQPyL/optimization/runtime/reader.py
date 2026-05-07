import json
import pickle
import sqlite3
from importlib import import_module

import numpy as np

from ...core.runtime import export_reader_summary, from_json_array
from ...core.runtime_reader import BaseReader
from ..population import Population


class OptReader(BaseReader):
    """
    Read optimization results from sqlite files.
    """
    @classmethod
    def list_runs(cls, result_dir):
        return super().list_runs(
            result_dir,
            run_columns="runId, algorithm, problem, status, finalFEs, finalIters, runtime, createdAt, finishedAt",
        )

    def __init__(self, dbPath):
        self.dbPath = str(dbPath)
        self.conn = sqlite3.connect(self.dbPath)
        self.conn.row_factory = sqlite3.Row

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
            method=run["algorithm"],
            problem_name=run["problem"],
            n_input=run["nInput"],
            n_output=run["nObj"],
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

    def load_algorithm(self):
        run = self.get_run()
        if run is None:
            raise ValueError("No run record found in sqlite database.")

        cls = self._resolveAlgorithmClass(run["algorithm"])
        params = self.get_run_params()
        kwargs = self._parseAlgorithmParams(params)
        return cls(**kwargs)

    def load_problem(self):
        row = self.conn.execute("SELECT problemPayload FROM run LIMIT 1").fetchone()
        if row is None or row["problemPayload"] is None:
            raise ValueError("No problem payload found in sqlite database.")
        return pickle.loads(row["problemPayload"])

    def list_snapshots(self):
        rows = self.conn.execute(
            """
            SELECT snapshotId, iter, fe, elapsed, bestObj, paretoSize, hypervolume, constraintViolation
            FROM snapshot
            ORDER BY snapshotId
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def load_population(self, snapshotId):
        return self._loadByRole(snapshotId, "population")

    def load_best(self, snapshotId):
        rows = self.conn.execute(
            "SELECT DISTINCT role FROM snapshotMember WHERE snapshotId = ? AND role IN ('best', 'pareto')",
            (snapshotId,),
        ).fetchall()
        roles = [row["role"] for row in rows]
        if "best" in roles:
            return self._loadByRole(snapshotId, "best")
        if "pareto" in roles:
            return self._loadByRole(snapshotId, "pareto")
        raise ValueError(f"No best/pareto records found for snapshotId={snapshotId}")

    def load_last_population(self):
        snapshotId = self._getLastSnapshotId()
        return self.load_population(snapshotId)

    def load_last_best(self):
        snapshotId = self._getLastSnapshotId()
        return self.load_best(snapshotId)

    def _getLastSnapshotId(self):
        row = self.conn.execute("SELECT snapshotId FROM snapshot ORDER BY snapshotId DESC LIMIT 1").fetchone()
        if row is None:
            raise ValueError("No snapshot found in sqlite database.")
        return row["snapshotId"]

    def _loadByRole(self, snapshotId, role):
        rows = self.conn.execute(
            """
            SELECT idx, decs, objs, cons, frontNo, crowdDis
            FROM snapshotMember
            WHERE snapshotId = ? AND role = ?
            ORDER BY idx
            """,
            (snapshotId, role),
        ).fetchall()
        if not rows:
            raise ValueError(f"No role={role!r} rows found for snapshotId={snapshotId}")

        decs = []
        objs = []
        cons = []
        hasObjs = True
        hasCons = False
        frontNo = []
        crowdDis = []

        for row in rows:
            decs.append(from_json_array(row["decs"]))
            obj = from_json_array(row["objs"])
            con = from_json_array(row["cons"])
            if obj is None:
                hasObjs = False
            else:
                objs.append(obj)
            if con is not None:
                hasCons = True
                cons.append(con)
            frontNo.append(row["frontNo"])
            crowdDis.append(row["crowdDis"])

        pop = Population(
            decs=np.vstack(decs),
            objs=np.vstack(objs) if hasObjs and objs else None,
            cons=np.vstack(cons) if hasCons and cons else None,
        )

        if any(value is not None for value in frontNo):
            pop.frontNo = np.asarray(frontNo, dtype=float)
        if any(value is not None for value in crowdDis):
            pop.crowdDis = np.asarray(crowdDis, dtype=float)
        return pop

    def _resolveAlgorithmClass(self, algorithmName):
        normalized = algorithmName.replace("-", "_").lower()
        candidates = [
            ("UQPyL.optimization.soea.ga", "GA", "ga"),
            ("UQPyL.optimization.soea.de", "DE", "de"),
            ("UQPyL.optimization.soea.csa", "CSA", "csa"),
            ("UQPyL.optimization.soea.sce_ua", "SCE_UA", "sce_ua"),
            ("UQPyL.optimization.soea.ml_sce_ua", "ML_SCE_UA", "ml_sce_ua"),
            ("UQPyL.optimization.soea.pso", "PSO", "pso"),
            ("UQPyL.optimization.soea.abc", "ABC", "abc"),
            ("UQPyL.optimization.expensive.asmo", "ASMO", "asmo"),
            ("UQPyL.optimization.expensive.ego", "EGO", "ego"),
            ("UQPyL.optimization.moea.nsga_ii", "NSGAII", "nsgaii"),
            ("UQPyL.optimization.moea.nsga_iii", "NSGAIII", "nsgaiii"),
            ("UQPyL.optimization.moea.moea_d", "MOEAD", "moea_d"),
            ("UQPyL.optimization.moea.rvea", "RVEA", "rvea"),
            ("UQPyL.optimization.expensive.moasmo", "MOASMO", "moasmo"),
        ]
        for moduleName, className, alias in candidates:
            if normalized in {className.lower(), alias}:
                module = import_module(moduleName)
                return getattr(module, className)
        raise ValueError(f"Unsupported algorithm name for reconstruction: {algorithmName}")

    def _parseAlgorithmParams(self, params):
        ignored = {
            "seed",
            "optType",
        }
        kwargs = {}
        for key, value in params.items():
            if key in ignored:
                continue
            kwargs[key] = self._parseValue(value)
        return kwargs

    def _parseValue(self, value):
        try:
            return eval(value, {"__builtins__": {}}, {})
        except Exception:
            return value
