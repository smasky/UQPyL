"""Runtime reader field names.

Migrated from test_review_a07_a15.py; original regression provenance is retained below.
"""

from contextlib import closing


# Regression source: test_review_a07_a15.py::testReaderListUsesSnakeCaseFields
def testReaderListUsesSnakeCaseFields(tmp_path):
    import sqlite3
    from UQPyL.core.runtime_reader import BaseReader

    path = tmp_path / "run.sqlite3"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TABLE run (runId TEXT, createdAt TEXT, finishedAt TEXT, finalFEs INTEGER, finalIters INTEGER)"
        )
        conn.execute("INSERT INTO run VALUES ('id','start','end',12,2)")
    row = BaseReader.list_runs(tmp_path, "*")[0]
    assert set(row) == {"run_id", "created_at", "finished_at", "final_fes", "final_iters", "db_path", "file_name"}
    assert row["final_fes"] == 12 and row["file_name"] == "run.sqlite3"
