import json
import sqlite3

import numpy as np

from ...core.runtime import array_to_json
from ...core.runtime_storage import BaseSqliteStorage


class SqliteStorage(BaseSqliteStorage):
    """
    Persist optimization runs and snapshots into sqlite files.
    """
    def _makeRunId(self, algorithmName, problemName):
        _, runId = self._db_path(algorithmName, problemName)
        return runId

    def _insert_run(self, conn, runId, obj, now):
        problem = obj.problem
        conn.execute(
            """
            INSERT INTO run (
                runId, algorithm, problem, seed, nInput, nObj, nCon,
                maxFEs, maxIters, verboseFreq, saveFreq,
                status, finalFEs, finalIters, runtime, createdAt, finishedAt,
                problemPayload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                runId,
                obj.name,
                problem.name,
                obj.get("seed"),
                problem.nInput,
                problem.nObj,
                problem.nCon,
                obj.maxFEs,
                obj.maxIter,
                obj.verboseFreq,
                obj.saveFreq,
                "running",
                0,
                0,
                0.0,
                now,
                None,
                self._problem_blob(problem),
            ),
        )

    def saveSnapshot(self, session, obj, result, isFinal=False):
        conn = session.conn
        runId = session.run_id
        problem = obj.problem

        bestObj = None
        paretoSize = None
        hypervolume = None
        if problem.nObj == 1 and result.bestObjs is not None:
            bestObj = float(result.bestObjs[0, 0])
        elif result.bestObjs is not None:
            paretoSize = int(result.bestObjs.shape[0])
            hypervolume = None if result.bestMetric is None else float(result.bestMetric)

        constraintViolation = 0.0
        if result.bestCons is not None:
            constraintViolation = float(np.sum(np.maximum(0.0, result.bestCons)))

        populationPayload = array_to_json({
            "decs": None if obj.state.currentPop is None else obj.state.currentPop.decs.tolist(),
            "objs": None if obj.state.currentPop is None or obj.state.currentPop.objs is None else obj.state.currentPop.objs.tolist(),
            "cons": None if obj.state.currentPop is None or obj.state.currentPop.cons is None else obj.state.currentPop.cons.tolist(),
        })
        bestPayload = array_to_json({
            "decs": None if result.bestDecs is None else result.bestDecs.tolist(),
            "objs": None if result.bestObjs is None else result.bestObjs.tolist(),
            "cons": None if result.bestCons is None else result.bestCons.tolist(),
        })

        cur = conn.execute(
            """
            INSERT INTO snapshot (
                runId, iter, fe, elapsed, bestObj, paretoSize, hypervolume,
                constraintViolation, populationPayload, bestPayload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                runId,
                result.iters,
                result.FEs,
                result.runtime,
                bestObj,
                paretoSize,
                hypervolume,
                constraintViolation,
                populationPayload,
                bestPayload,
            ),
        )
        snapshotId = cur.lastrowid

        currentPop = obj.state.currentPop
        if currentPop is not None:
            self._insertMembers(conn, snapshotId, "population", currentPop)

        if result.bestDecs is not None and result.bestObjs is not None:
            role = "best" if problem.nObj == 1 else "pareto"
            self._insertBestMembers(conn, snapshotId, role, result)

        if isFinal:
            self.finalize_run(
                session,
                status="finished",
                runtime=result.runtime,
                final_fes=result.FEs,
                final_iters=result.iters,
            )

        conn.commit()

    def _insertMembers(self, conn, snapshotId, role, pop):
        frontNo = pop.frontNo if pop.frontNo is not None else np.full((len(pop),), np.nan)
        crowdDis = pop.crowdDis if pop.crowdDis is not None else np.full((len(pop),), np.nan)

        for idx in range(len(pop)):
            conn.execute(
                """
                INSERT INTO snapshotMember (
                    snapshotId, idx, role, decs, objs, cons, frontNo, crowdDis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshotId,
                    idx,
                    role,
                    array_to_json(pop.decs[idx]),
                    array_to_json(None if pop.objs is None else pop.objs[idx]),
                    array_to_json(None if pop.cons is None else pop.cons[idx]),
                    None if np.isnan(frontNo[idx]) else float(frontNo[idx]),
                    None if np.isnan(crowdDis[idx]) else float(crowdDis[idx]),
                ),
            )

    def _insertBestMembers(self, conn, snapshotId, role, result):
        n = result.bestDecs.shape[0]
        bestCons = result.bestCons
        for idx in range(n):
            conn.execute(
                """
                INSERT INTO snapshotMember (
                    snapshotId, idx, role, decs, objs, cons, frontNo, crowdDis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshotId,
                    idx,
                    role,
                    array_to_json(result.bestDecs[idx]),
                    array_to_json(result.bestObjs[idx]),
                    array_to_json(None if bestCons is None else bestCons[idx]),
                    None,
                    None,
                ),
            )

    def _create_schema(self, conn):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run (
                runId TEXT PRIMARY KEY,
                algorithm TEXT NOT NULL,
                problem TEXT NOT NULL,
                seed INTEGER,
                nInput INTEGER NOT NULL,
                nObj INTEGER NOT NULL,
                nCon INTEGER NOT NULL,
                maxFEs INTEGER,
                maxIters INTEGER,
                verboseFreq INTEGER,
                saveFreq INTEGER,
                status TEXT NOT NULL,
                finalFEs INTEGER,
                finalIters INTEGER,
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

            CREATE TABLE IF NOT EXISTS snapshot (
                snapshotId INTEGER PRIMARY KEY AUTOINCREMENT,
                runId TEXT NOT NULL,
                iter INTEGER NOT NULL,
                fe INTEGER NOT NULL,
                elapsed REAL,
                bestObj REAL,
                paretoSize INTEGER,
                hypervolume REAL,
                constraintViolation REAL,
                populationPayload TEXT,
                bestPayload TEXT,
                FOREIGN KEY(runId) REFERENCES run(runId)
            );

            CREATE TABLE IF NOT EXISTS snapshotMember (
                snapshotId INTEGER NOT NULL,
                idx INTEGER NOT NULL,
                role TEXT NOT NULL,
                decs TEXT,
                objs TEXT,
                cons TEXT,
                frontNo REAL,
                crowdDis REAL,
                FOREIGN KEY(snapshotId) REFERENCES snapshot(snapshotId)
            );
            """
        )
