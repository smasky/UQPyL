import json
import ast
import inspect
import warnings
import pickle
import sqlite3
from importlib import import_module

import numpy as np

from ...core.runtime import export_reader_summary, from_json_array
from ...core.runtime_reader import BaseReader
from ..population import Population
from .result import OptHistory, OptResult


class OptReader(BaseReader):
    domain = 'optimization'
    """
    Read optimization results from sqlite files.
    """
    @classmethod
    def list_runs(cls, result_dir):
        return super().list_runs(
            result_dir,
            run_columns="runId, algorithm, problem, status, finalFEs, finalIters, runtime, createdAt, finishedAt",
        )





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
        missing = kwargs.pop('_unrestored_components', [])
        for key in ('maxFEs', 'maxIters', 'verboseFreq', 'saveFreq'):
            kwargs.setdefault(key, run[key])
        accepted = inspect.signature(cls).parameters
        algorithm = cls(**{key: value for key, value in kwargs.items() if key in accepted})
        for key in ('tolerate', 'maxTolerates', 'hvRefPoint'):
            if key in kwargs and key not in accepted:
                setattr(algorithm, key, kwargs[key])
        unknown = set(kwargs) - set(accepted) - {'tolerate', 'maxTolerates', 'hvRefPoint'}
        if missing or unknown:
            warnings.warn(f"Configuration restored; restore components/settings manually: {sorted(set(missing) | unknown)}. "
                          "This does not resume optimizer state.", UserWarning, stacklevel=2)
        return algorithm

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
        return self._loadByRole(snapshotId, "pareto")

    def load_candidates(self, snapshotId):
        """Load infeasible diagnostics; never substitutes them for the Pareto front."""
        return self._loadByRole(snapshotId, "candidate")

    def load_last_candidates(self):
        return self.load_candidates(self._getLastSnapshotId())

    def load_last_population(self):
        snapshotId = self._getLastSnapshotId()
        return self.load_population(snapshotId)

    def load_last_best(self):
        snapshotId = self._getLastSnapshotId()
        return self.load_best(snapshotId)

    def load_result(self):
        """Restore the result with the history available in saved snapshots."""
        snapshots = self.list_snapshots()
        if not snapshots:
            raise ValueError("No snapshot found in sqlite database.")
        # Final saves may repeat the last completed iteration.
        snapshots = list({(row['iter'], row['fe']): row for row in snapshots}.values())
        history = OptHistory()
        for row in snapshots:
            history.iterToFEs.append([row['iter'], row['fe']])
            history.metrics.append(row['hypervolume'])
            history.bestObjHistory.append(row['bestObj'])
            history.bestMetricHistory.append(row['hypervolume'])
            history.numBestHistory.append(row['paretoSize'])
            population = self.load_population(row['snapshotId'])
            best = self.load_best(row['snapshotId'])
            candidates = self.load_candidates(row['snapshotId'])
            payload = json.loads(self.conn.execute('SELECT bestPayload FROM snapshot WHERE snapshotId=?',
                                                  (row['snapshotId'],)).fetchone()[0])
            history.snapshotIterToFEs.append([row['iter'], row['fe']])
            history.populations.append(dict(decs=population.decs, objs=population.objs,
                                            cons=population.cons, constraint_weights=population.conWgt))
            history.bests.append(dict(bestDecs=best.decs, bestObjs=best.objs, bestCons=best.cons,
                                      bestFeasible=bool(payload['best_feasible']),
                                      candidateDecs=candidates.decs if len(candidates) else None,
                                      candidateObjs=candidates.objs if len(candidates) else None,
                                      candidateCons=candidates.cons if len(candidates) else None,
                                      minViolation=payload.get('min_violation')))
            history.improvedHistory.append(payload.get('improved'))
        last = snapshots[-1]
        return OptResult(
            bestDecs=best.decs, bestObjs=best.objs, bestCons=best.cons,
            bestMetric=last['hypervolume'], bestFeasible=bool(payload['best_feasible']),
            appearFEs=payload.get('appear_fes'), appearIters=payload.get('appear_iters'),
            FEs=last['fe'], iters=last['iter'], runtime=last['elapsed'], history=history,
            candidateDecs=candidates.decs if len(candidates) else None,
            candidateObjs=candidates.objs if len(candidates) else None,
            candidateCons=candidates.cons if len(candidates) else None,
            minViolation=payload.get('min_violation'),
            extra={key: payload.get(key) for key in ('constraint_weights', 'hv_reference_point')},
        )

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
            snapshot = self.conn.execute("SELECT bestPayload FROM snapshot WHERE snapshotId = ?",
                                         (snapshotId,)).fetchone()
            if snapshot is None or role not in ("pareto", "candidate"):
                raise ValueError(f"No role={role!r} rows found for snapshotId={snapshotId}")
            payload = json.loads(snapshot[0]) if snapshot[0] else {}
            run = self.get_run()
            return Population(np.empty((0, run["nInput"])), np.empty((0, run["nObj"])),
                              np.empty((0, run["nCon"])) if run["nCon"] else None,
                              payload.get("constraint_weights"))

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

        payloadColumn = "populationPayload" if role == "population" else "bestPayload"
        snapshot = self.conn.execute(
            f"SELECT {payloadColumn} FROM snapshot WHERE snapshotId = ?", (snapshotId,)
        ).fetchone()
        payload = {} if snapshot is None or snapshot[0] is None else json.loads(snapshot[0])
        pop = Population(
            conWgt=payload.get("constraint_weights"),
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
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
