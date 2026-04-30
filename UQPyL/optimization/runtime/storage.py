import json
import os
import sqlite3
import uuid
from datetime import datetime

import numpy as np
import pickle


def _to_json_array(value):
    if value is None:
        return None
    arr = np.asarray(value)
    return json.dumps(arr.tolist(), ensure_ascii=True)


class SqliteStorage:
    """
    Persist optimization runs and snapshots into sqlite files.
    """
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.resultDir = os.path.join(rootDir, "Result")
        os.makedirs(self.resultDir, exist_ok=True)

    def _dbPath(self, algorithmName, problemName):
        runId = self._makeRunId(algorithmName)
        filename = f"{runId}.sqlite3"
        return os.path.join(self.resultDir, filename), runId

    def _makeRunId(self, algorithmName):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = uuid.uuid4().hex[:4]
        return f"{algorithmName.lower()}_{timestamp}_{suffix}"

    def createRun(self, obj):
        problem = obj.problem
        dbPath, runId = self._dbPath(obj.name, problem.name)
        conn = sqlite3.connect(dbPath)
        self._createSchema(conn)
        now = datetime.now().isoformat(timespec="seconds")

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
                obj.getParaVal("seed"),
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

        populationPayload = _to_json_array({
            "decs": None if obj.state.currentPop is None else obj.state.currentPop.decs.tolist(),
            "objs": None if obj.state.currentPop is None or obj.state.currentPop.objs is None else obj.state.currentPop.objs.tolist(),
            "cons": None if obj.state.currentPop is None or obj.state.currentPop.cons is None else obj.state.currentPop.cons.tolist(),
        })
        bestPayload = _to_json_array({
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

    def close(self, storageCtx):
        storageCtx["conn"].close()

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
                    _to_json_array(pop.decs[idx]),
                    _to_json_array(None if pop.objs is None else pop.objs[idx]),
                    _to_json_array(None if pop.cons is None else pop.cons[idx]),
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
                    _to_json_array(result.bestDecs[idx]),
                    _to_json_array(result.bestObjs[idx]),
                    _to_json_array(None if bestCons is None else bestCons[idx]),
                    None,
                    None,
                ),
            )

    def _createSchema(self, conn):
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
