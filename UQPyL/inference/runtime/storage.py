import json
import pickle

import numpy as np

from ...core.runtime import array_to_json
from ...core.runtime_storage import BaseSqliteStorage


class SqliteStorage(BaseSqliteStorage):
    def _makeRunId(self, methodName, problemName):
        _, runId = self._db_path(methodName, problemName)
        return runId

    def create_run(self, obj):
        session = super().create_run(obj)
        session.conn.execute("PRAGMA journal_mode=MEMORY")
        return session

    def _insert_run(self, conn, runId, obj, now):
        problem = obj.problem
        conn.execute(
            """
            INSERT INTO run (
                runId, method, problem, seed, nInput, nOutput, nCon, nChains,
                maxIters, warmUp, verboseFreq, saveFreq, status, finalFEs,
                finalIters, runtime, createdAt, finishedAt, problemPayload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                runId,
                obj.name,
                problem.name,
                obj.get("seed"),
                problem.nInput,
                problem.nOutput,
                problem.nCons,
                obj.get("nChains") if "nChains" in obj.params.data else None,
                obj.maxIters,
                obj.get("warmUp") if "warmUp" in obj.params.data else None,
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
        state = obj.state

        cur = conn.execute(
            """
            INSERT INTO snapshot (
                runId, iter, fe, elapsed, meanLogProb, bestObj, feasibleRate,
                acceptanceRateMean, statePayload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                runId,
                result.iters,
                result.FEs,
                result.runtime,
                state.meanLogProb,
                state.bestObj,
                state.feasibleRate,
                state.acceptanceRateMean,
                json.dumps(state.buildSnapshot(result.FEs, result.iters), ensure_ascii=True),
            ),
        )
        snapshotId = cur.lastrowid
        self._insertSnapshotMembers(conn, snapshotId, state)

        if isFinal:
            self.finalize_run(
                session,
                status="finished",
                runtime=result.runtime,
                final_fes=result.FEs,
                final_iters=result.iters,
            )

        conn.commit()

    def saveResultArtifact(self, session, result):
        conn = session.conn
        runId = session.run_id
        conn.execute(
            "INSERT INTO artifact (runId, name, payload) VALUES (?, ?, ?)",
            (
                runId,
                "result",
                self._problem_blob(result),
            ),
        )
        conn.commit()

    def _insertSnapshotMembers(self, conn, snapshotId, state):
        if state.decs is None or state.decs.shape[1] == 0:
            return
        last = state.decs.shape[1] - 1
        nChains = state.decs.shape[0]
        for chain in range(nChains):
            conn.execute(
                """
                INSERT INTO snapshotMember (
                    snapshotId, chain, decs, objs, cons, logProb, accepted, feasible
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshotId,
                    chain,
                    array_to_json(state.decs[chain, last]),
                    array_to_json(state.objs[chain, last]),
                    array_to_json(None if state.cons is None else state.cons[chain, last]),
                    None if state.logProb is None else float(state.logProb[chain, last]),
                    int(state.accepted[chain, last]),
                    int(state.feasibleMask[chain, last]),
                ),
            )

    def _create_schema(self, conn):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run (
                runId TEXT PRIMARY KEY,
                method TEXT NOT NULL,
                problem TEXT NOT NULL,
                seed INTEGER,
                nInput INTEGER NOT NULL,
                nOutput INTEGER NOT NULL,
                nCon INTEGER NOT NULL,
                nChains INTEGER,
                maxIters INTEGER,
                warmUp INTEGER,
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
                meanLogProb REAL,
                bestObj REAL,
                feasibleRate REAL,
                acceptanceRateMean REAL,
                statePayload TEXT,
                FOREIGN KEY(runId) REFERENCES run(runId)
            );

            CREATE TABLE IF NOT EXISTS snapshotMember (
                snapshotId INTEGER NOT NULL,
                chain INTEGER NOT NULL,
                decs TEXT,
                objs TEXT,
                cons TEXT,
                logProb REAL,
                accepted INTEGER,
                feasible INTEGER,
                FOREIGN KEY(snapshotId) REFERENCES snapshot(snapshotId)
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
