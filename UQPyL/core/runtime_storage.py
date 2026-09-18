from __future__ import annotations

import sqlite3
from datetime import datetime

from .runtime import build_db_path, pickle_to_blob
from .runtime_session import RunSession
from .runtime_lifecycle import recordCleanupError


class BaseSqliteStorage:
    domain = None
    def __init__(self, root_dir):
        self.root_dir = str(root_dir)

    def _db_path(self, method_name, problem_name, run_id=None):
        return build_db_path(self.root_dir, method_name, problem_name, run_id)

    def create_run(self, obj):
        if not hasattr(self, "_create_schema") or self.__class__._create_schema is BaseSqliteStorage._create_schema:
            raise NotImplementedError("Subclasses must implement _create_schema().")
        if not hasattr(self, "_insert_run") or self.__class__._insert_run is BaseSqliteStorage._insert_run:
            raise NotImplementedError("Subclasses must implement _insert_run().")

        db_path, run_id = self._db_path(obj.name, obj.problem.name, getattr(obj, 'runId', None))
        conn = sqlite3.connect(db_path)
        session = RunSession(run_id=run_id, db_path=db_path, conn=conn, root_dir=self.root_dir)
        ownsRun = False
        try:
            self._configure_connection(conn)
            self._create_schema(conn)
            if self.domain is not None:
                conn.execute("CREATE TABLE IF NOT EXISTS runtimeMeta (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute("INSERT OR IGNORE INTO runtimeMeta VALUES ('domain', ?)", (self.domain,))
                if conn.execute("SELECT value FROM runtimeMeta WHERE name='domain'").fetchone()[0] != self.domain:
                    raise ValueError("Database belongs to a different runtime domain.")
            self._insert_run(conn, run_id, obj, datetime.now().isoformat(timespec="seconds"))
            conn.commit()  # Keep an identifiable run if parameter initialization fails.
            ownsRun = True
            self._save_params(conn, run_id, obj)
            conn.commit()
            return session
        except BaseException as error:
            self.abort_run(session, error, owns_run=ownsRun)
            raise

    def _configure_connection(self, conn):
        pass

    def abort_run(self, session, error, *, runtime=0.0, final_fes=None, final_iters=None, owns_run=True):
        """Best-effort rollback, failure status and close; preserve the cause."""
        try:
            session.conn.rollback()
        except BaseException as cleanupError:
            recordCleanupError(error, "rollback", cleanupError)
        else:
            # Never commit failure metadata over an unrolled-back partial write.
            try:
                if owns_run:
                    self.finalize_run(session, status="failed", runtime=runtime,
                                      final_fes=final_fes, final_iters=final_iters)
            except BaseException as cleanupError:
                recordCleanupError(error, "failure status", cleanupError)
        try:
            self.close(session)
        except BaseException as cleanupError:
            recordCleanupError(error, "close", cleanupError)

    def finalize_run(self, session: RunSession, *, status="finished", runtime=0.0, final_fes=None, final_iters=None):
        finished_at = datetime.now().isoformat(timespec="seconds")
        columns = ["status=?", "runtime=?", "finishedAt=?"]
        values = [status, runtime, finished_at]
        if final_fes is not None:
            columns.append("finalFEs=?")
            values.append(final_fes)
        if final_iters is not None:
            columns.append("finalIters=?")
            values.append(final_iters)
        values.append(session.run_id)
        session.conn.execute(f"UPDATE run SET {', '.join(columns)} WHERE runId=?", values)
        session.conn.commit()

    def close(self, session: RunSession):
        if session.conn is not None:
            session.conn.close()
            session.conn = None

    def _save_params(self, conn, run_id, obj):
        params = getattr(obj, "params", None) or getattr(obj, "setting", None)
        if params is None:
            return
        for name, value in params.items():
            conn.execute(
                "INSERT INTO runParam (runId, name, value) VALUES (?, ?, ?)",
                (run_id, name, repr(value)),
            )

    def _problem_blob(self, problem):
        blob = pickle_to_blob(problem)
        return sqlite3.Binary(blob) if blob is not None else None

    def _create_schema(self, conn):
        raise NotImplementedError

    def _insert_run(self, conn, run_id, obj, now):
        raise NotImplementedError
