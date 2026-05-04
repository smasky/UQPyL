import json
import os
import pickle
import re
import sqlite3
import uuid
from datetime import datetime

import numpy as np


def _to_json_array(value):
    if value is None:
        return None
    return json.dumps(np.asarray(value).tolist(), ensure_ascii=True)


def _slugify_name(name):
    text = str(name).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "problem"


class SqliteStorage:
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.resultDir = os.path.join(rootDir, "Result")
        os.makedirs(self.resultDir, exist_ok=True)

    def _makeRunId(self, methodName, problemName):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = uuid.uuid4().hex[:4]
        problemSlug = _slugify_name(problemName)
        return f"{methodName.lower()}_{problemSlug}_{timestamp}_{suffix}"

    def _dbPath(self, methodName, problemName):
        runId = self._makeRunId(methodName, problemName)
        return os.path.join(self.resultDir, f"{runId}.sqlite3"), runId

    def createRun(self, obj):
        problem = obj.problem
        dbPath, runId = self._dbPath(obj.name, problem.name)
        conn = sqlite3.connect(dbPath)
        conn.execute("PRAGMA journal_mode=MEMORY")
        self._createSchema(conn)
        now = datetime.now().isoformat(timespec="seconds")

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
                obj.getParaVal("seed"),
                problem.nInput,
                problem.nOutput,
                problem.nCons,
                obj.getParaVal("nChains") if "nChains" in obj.params.data else None,
                obj.maxIters,
                obj.getParaVal("warmUp") if "warmUp" in obj.params.data else None,
                obj.verboseFreq,
                obj.saveFreq,
                "running",
                0,
                0,
                0.0,
                now,
                None,
                sqlite3.Binary(pickle.dumps(problem, protocol=pickle.HIGHEST_PROTOCOL)),
            ),
        )

        for name, value in obj.params.items():
            conn.execute(
                "INSERT INTO runParam (runId, name, value) VALUES (?, ?, ?)",
                (runId, name, repr(value)),
            )

        conn.commit()
        return {"conn": conn, "dbPath": dbPath, "runId": runId}

    def saveSnapshot(self, storageCtx, obj, result, isFinal=False):
        conn = storageCtx["conn"]
        runId = storageCtx["runId"]
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
            finishedAt = datetime.now().isoformat(timespec="seconds")
            conn.execute(
                """
                UPDATE run
                SET status=?, finalFEs=?, finalIters=?, runtime=?, finishedAt=?
                WHERE runId=?
                """,
                ("finished", result.FEs, result.iters, result.runtime, finishedAt, runId),
            )

        conn.commit()

    def saveResultArtifact(self, storageCtx, result):
        conn = storageCtx["conn"]
        runId = storageCtx["runId"]
        conn.execute(
            "INSERT INTO artifact (runId, name, payload) VALUES (?, ?, ?)",
            (
                runId,
                "result",
                sqlite3.Binary(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)),
            ),
        )
        conn.commit()

    def close(self, storageCtx):
        storageCtx["conn"].close()

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
                    _to_json_array(state.decs[chain, last]),
                    _to_json_array(state.objs[chain, last]),
                    _to_json_array(None if state.cons is None else state.cons[chain, last]),
                    None if state.logProb is None else float(state.logProb[chain, last]),
                    int(state.accepted[chain, last]),
                    int(state.feasibleMask[chain, last]),
                ),
            )

    def _createSchema(self, conn):
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
