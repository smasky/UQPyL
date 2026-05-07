import sqlite3
import uuid
from pathlib import Path

import pytest


def test_run_session_holds_runtime_resources():
    from UQPyL.core.runtime_session import RunSession

    conn = sqlite3.connect(":memory:")
    session = RunSession(
        run_id="demo_run",
        db_path="D:/UQ/Result/demo_run.sqlite3",
        conn=conn,
        root_dir="D:/UQ",
    )

    assert session.run_id == "demo_run"
    assert session.db_path.endswith("demo_run.sqlite3")
    assert session.conn is conn
    assert session.root_dir == "D:/UQ"

    conn.close()


@pytest.fixture
def local_tmp_path():
    path = Path("D:/UQ/Result/_pytest_runtime_common") / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        # Windows CI/local runs may keep sqlite handles briefly; leave cleanup to gitignore.
        pass


def test_base_sqlite_reader_lists_runs_from_result_dir(local_tmp_path):
    from UQPyL.core.runtime_reader import BaseReader

    result_dir = local_tmp_path / "Result"
    result_dir.mkdir()
    db_path = result_dir / "demo.sqlite3"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE run (
            runId TEXT PRIMARY KEY,
            method TEXT NOT NULL,
            problem TEXT NOT NULL,
            status TEXT NOT NULL,
            runtime REAL,
            createdAt TEXT NOT NULL,
            finishedAt TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO run (runId, method, problem, status, runtime, createdAt, finishedAt)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("demo_001", "DemoMethod", "DemoProblem", "finished", 1.25, "2026-05-07T10:00:00", "2026-05-07T10:00:01"),
    )
    conn.commit()
    conn.close()

    runs = BaseReader.list_runs(
        local_tmp_path,
        run_columns="runId, method, problem, status, runtime, createdAt, finishedAt",
    )

    assert len(runs) == 1
    assert runs[0]["run_id"] == "demo_001"
    assert runs[0]["dbPath"].endswith("demo.sqlite3")


def test_base_sqlite_storage_requires_schema_hooks(local_tmp_path):
    from UQPyL.core.runtime_storage import BaseSqliteStorage

    storage = BaseSqliteStorage(local_tmp_path)

    with pytest.raises(NotImplementedError):
        storage.create_run(object())
