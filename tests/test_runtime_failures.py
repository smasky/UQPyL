import sqlite3

import numpy as np
import pytest

from UQPyL.analysis import RSA
from UQPyL.calibration import GLUE
from UQPyL.core.runtime_storage import BaseSqliteStorage
from UQPyL.inference import MH
from UQPyL.optimization.soea import GA
from UQPyL.problem import ModelProblem, Problem
from UQPyL.analysis.runtime import AnaReader
from UQPyL.calibration import CalReader
from UQPyL.inference import InfReader
from UQPyL.optimization.runtime import OptReader


def makeRun(kind, tmpPath, *, saveFlag=True):
    control = {"error": None}

    def objective(X):
        if control["error"] is not None:
            raise control["error"]
        return np.sum(X ** 2, axis=1, keepdims=True) + 1

    def simulate(X):
        return np.repeat(objective(X)[:, :, None], 3, axis=1)

    flags = dict(verboseFlag=False, saveFlag=saveFlag)
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=objective)
    samples = np.random.default_rng(12).random((30, 2))
    if kind == "optimization":
        method = GA(nPop=4, maxFEs=12, saveFreq=1, **flags)
        run = lambda: method.run(problem, seed=12)
    elif kind == "inference":
        method = MH(nChains=2, warmUp=0, maxIters=3, saveFreq=1, **flags)
        run = lambda: method.run(problem, seed=12)
    elif kind == "analysis":
        method = RSA(nRegion=2, **flags)
        run = lambda: method.analyze(problem, samples)
    else:
        problem = ModelProblem(nInput=2, lb=0, ub=1, simFunc=simulate, obs=np.ones((3, 1)))
        method = GLUE(**flags)
        run = lambda: method.run(problem, samples, threshold=10)
    problem.workDir = str(tmpPath)
    return method, run, control


@pytest.fixture
def sessions(monkeypatch):
    captured = []
    createRun = BaseSqliteStorage.create_run

    def trackSession(self, obj):
        session = createRun(self, obj)
        captured.append((session, session.conn))
        return session

    monkeypatch.setattr(BaseSqliteStorage, "create_run", trackSession)
    return captured


def assertClosed(method, session, conn, status):
    assert method.session is None
    assert session.conn is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    with sqlite3.connect(session.db_path) as reader:
        row = reader.execute("SELECT status, finishedAt, runtime FROM run").fetchone()
    assert row[0] == status and row[1] is not None and row[2] >= 0


@pytest.mark.parametrize("kind", ["optimization", "inference", "analysis", "calibration"])
@pytest.mark.parametrize("stage", ["model", "setup", "save", "after_finalize"])
def test_failure_closes_marks_failed_and_allows_reuse(kind, stage, tmp_path, monkeypatch, sessions):
    method, run, control = makeRun(kind, tmp_path)
    error = RuntimeError(f"injected {stage} failure")
    with monkeypatch.context() as patch:
        if stage == "model":
            control["error"] = error
        elif stage == "setup":
            setup = method.setup

            def failSetup(*args, **kwargs):
                setup(*args, **kwargs)
                raise error

            patch.setattr(method, "setup", failSetup)
        elif stage == "save":
            baseModule = __import__(f"UQPyL.{kind}.base", fromlist=["SqliteStorage"])

            def failSave(self, session, *args, **kwargs):
                session.conn.execute("INSERT INTO runParam VALUES (?, ?, ?)", (session.run_id, "partial_write", "1"))
                raise error

            patch.setattr(baseModule.SqliteStorage, "saveSnapshot" if kind in ("optimization", "inference") else "saveResult", failSave)
        else:
            finalize = method.finalize

            def failFinalize():
                finalize()
                raise error

            patch.setattr(method, "finalize", failFinalize)
        with pytest.raises(RuntimeError) as raised:
            run()
        assert raised.value is error
    control["error"] = None
    session, conn = sessions[-1]
    assertClosed(method, session, conn, "failed")
    readerClass = dict(optimization=OptReader, inference=InfReader,
                       analysis=AnaReader, calibration=CalReader)[kind]
    reader = readerClass(session.db_path)
    try:
        assert reader.get_run_summary()["status"] == "failed"
    finally:
        reader.close()
    with sqlite3.connect(session.db_path) as reader:
        assert reader.execute("SELECT count(*) FROM runParam WHERE name='partial_write'").fetchone()[0] == 0
    result = run()
    assert result is not None
    assert sessions[-1][0].run_id != session.run_id
    assertClosed(method, *sessions[-1], "finished")


@pytest.mark.parametrize("kind", ["optimization", "inference", "analysis", "calibration"])
def test_parameter_initialization_failure_closes_unreturned_session(kind, tmp_path, monkeypatch):
    method, run, _ = makeRun(kind, tmp_path)
    captured = []
    error = ValueError("parameter serialization failed")

    def failParams(self, conn, runId, obj):
        captured.append(conn)
        conn.execute("INSERT INTO runParam VALUES (?, ?, ?)", (runId, "partial_write", "1"))
        raise error

    monkeypatch.setattr(BaseSqliteStorage, "_save_params", failParams)
    with pytest.raises(ValueError) as raised:
        run()
    assert raised.value is error and method.session is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        captured[0].execute("SELECT 1")
    dbPath = next(tmp_path.rglob("*.sqlite3"))
    with sqlite3.connect(dbPath) as reader:
        assert reader.execute("SELECT status FROM run").fetchone()[0] == "failed"
        assert reader.execute("SELECT count(*) FROM runParam").fetchone()[0] == 0


@pytest.mark.parametrize("kind", ["optimization", "inference", "analysis", "calibration"])
def test_failure_without_storage_preserves_exception(kind, tmp_path):
    method, run, control = makeRun(kind, tmp_path, saveFlag=False)
    control["error"] = ValueError("model failed")
    with pytest.raises(ValueError) as raised:
        run()
    assert raised.value is control["error"]
    assert method.session is None and not list(tmp_path.rglob("*.sqlite3"))


def test_cleanup_error_does_not_replace_model_error(tmp_path, monkeypatch, sessions):
    method, run, control = makeRun("optimization", tmp_path)
    control["error"] = RuntimeError("original model error")

    def failStatus(*args, **kwargs):
        raise sqlite3.OperationalError("status write unavailable")

    monkeypatch.setattr(BaseSqliteStorage, "finalize_run", failStatus)
    with pytest.raises(RuntimeError) as raised:
        run()
    assert raised.value is control["error"]
    assert any("status write unavailable" in note for note in raised.value.__notes__)
    assert method.session is None and sessions[-1][0].conn is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        sessions[-1][1].execute("SELECT 1")


@pytest.mark.parametrize("kind", ["optimization", "inference"])
def test_failed_snapshot_rolls_back_only_current_write(kind, tmp_path, monkeypatch, sessions):
    method, run, _ = makeRun(kind, tmp_path)
    baseModule = __import__(f"UQPyL.{kind}.base", fromlist=["SqliteStorage"])
    saveSnapshot = baseModule.SqliteStorage.saveSnapshot
    count = 0

    def failSecondSave(self, session, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            session.conn.execute("INSERT INTO snapshot (runId, iter, fe) VALUES (?, ?, ?)", (session.run_id, 99, 99))
            raise RuntimeError("partial snapshot")
        return saveSnapshot(self, session, *args, **kwargs)

    monkeypatch.setattr(baseModule.SqliteStorage, "saveSnapshot", failSecondSave)
    with pytest.raises(RuntimeError, match="partial snapshot"):
        run()
    assertClosed(method, *sessions[-1], "failed")
    with sqlite3.connect(sessions[-1][0].db_path) as reader:
        assert reader.execute("SELECT count(*) FROM snapshot").fetchone()[0] == 1
        counters = reader.execute("SELECT finalFEs, finalIters FROM run").fetchone()
        assert counters == (method.FEs, method.iters)
    reader = (OptReader if kind == "optimization" else InfReader)(sessions[-1][0].db_path)
    try:
        assert len(reader.list_snapshots()) == 1
        if kind == "optimization":
            assert reader.load_last_population() is not None
        else:
            assert len(reader.load_last_snapshot_members()) == method.get("nChains")
    finally:
        reader.close()


@pytest.mark.parametrize("kind", ["analysis", "calibration"])
def test_result_serialization_failure_rolls_back_partial_artifacts(kind, tmp_path, monkeypatch, sessions):
    method, run, _ = makeRun(kind, tmp_path)
    makeBlob = BaseSqliteStorage._problem_blob

    def failArrayBlob(self, value):
        if isinstance(value, np.ndarray) or (kind == "calibration" and isinstance(value, dict) and "best_score" in value):
            raise ValueError("array serialization failed")
        return makeBlob(self, value)

    monkeypatch.setattr(BaseSqliteStorage, "_problem_blob", failArrayBlob)
    with pytest.raises(ValueError, match="array serialization failed"):
        run()
    assertClosed(method, *sessions[-1], "failed")
    with sqlite3.connect(sessions[-1][0].db_path) as reader:
        assert reader.execute("SELECT count(*) FROM artifact").fetchone()[0] == 0
        if kind == "analysis":
            assert reader.execute("SELECT count(*) FROM metric").fetchone()[0] == 0


@pytest.mark.parametrize("stage", ["_configure_connection", "_create_schema", "_insert_run"])
def test_early_database_initialization_failure_closes_connection(stage, tmp_path, monkeypatch):
    method, run, _ = makeRun("optimization", tmp_path)
    from UQPyL.optimization.runtime.storage import SqliteStorage
    captured = []

    def fail(self, conn, *args):
        captured.append(conn)
        raise RuntimeError("database initialization failed")

    monkeypatch.setattr(SqliteStorage, stage, fail)
    with pytest.raises(RuntimeError, match="database initialization failed"):
        run()
    assert method.session is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        captured[0].execute("SELECT 1")


def test_keyboard_interrupt_also_closes_session(tmp_path, sessions):
    method, run, control = makeRun("optimization", tmp_path)
    control["error"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as raised:
        run()
    assert raised.value is control["error"]
    assertClosed(method, *sessions[-1], "failed")


def test_subclass_super_run_keeps_outer_lifetime(tmp_path, sessions):
    class PostProcessGA(GA):
        def run(self, *args, **kwargs):
            super().run(*args, **kwargs)
            assert self.session.conn is not None
            raise RuntimeError("postprocessing failed")

    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda X: X.sum(axis=1))
    problem.workDir = str(tmp_path)
    method = PostProcessGA(nPop=4, maxFEs=4, verboseFlag=False, saveFlag=True)
    with pytest.raises(RuntimeError, match="postprocessing failed"):
        method.run(problem, seed=12)
    assertClosed(method, *sessions[-1], "failed")


def test_close_failure_is_reported_and_retains_session_for_cleanup(tmp_path, monkeypatch, sessions):
    method, run, control = makeRun("optimization", tmp_path)
    control["error"] = RuntimeError("original error")
    close = BaseSqliteStorage.close

    def failClose(self, session):
        raise OSError("close failed")

    monkeypatch.setattr(BaseSqliteStorage, "close", failClose)
    with pytest.raises(RuntimeError) as raised:
        run()
    assert raised.value is control["error"]
    assert any("close failed" in note for note in raised.value.__notes__)
    assert method.session is sessions[-1][0]
    with pytest.raises(RuntimeError, match="existing run session"):
        run()
    close(method.storage, method.session)
    method.session = None


def test_rollback_failure_does_not_commit_partial_data(tmp_path, sessions):
    method, run, control = makeRun("optimization", tmp_path)
    setup = method.setup

    class BrokenRollback:
        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def rollback(self):
            raise sqlite3.OperationalError("rollback failed")

    def setupWithPartialWrite(*args, **kwargs):
        setup(*args, **kwargs)
        session = method.session
        session.conn.execute("INSERT INTO runParam VALUES (?, ?, ?)", (session.run_id, "partial_write", "1"))
        session.conn = BrokenRollback(session.conn)

    method.setup = setupWithPartialWrite
    control["error"] = RuntimeError("original error")
    with pytest.raises(RuntimeError) as raised:
        run()
    assert raised.value is control["error"]
    assert any("rollback failed" in note for note in raised.value.__notes__)
    assert method.session is None
    with sqlite3.connect(sessions[-1][0].db_path) as reader:
        assert reader.execute("SELECT count(*) FROM runParam WHERE name='partial_write'").fetchone()[0] == 0
