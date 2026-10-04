"""Runtime domain and identity.

Migrated from test_remaining_review.py; original regression provenance is retained below.
"""

from contextlib import closing
import pytest


# Regression source: test_remaining_review.py::testReadersRejectOtherDomains
def testReadersRejectOtherDomains(tmp_path):
    from UQPyL.problem import Sphere
    from UQPyL.optimization.soea import GA
    from UQPyL.calibration import CalReader
    from UQPyL.inference import InfReader
    from UQPyL.analysis.runtime import AnaReader
    from UQPyL.optimization.runtime import OptReader

    problem = Sphere(nInput=2)
    problem.workDir = str(tmp_path)
    GA(nPop=4, maxIters=0, saveFlag=True, logFlag=False, verboseFlag=False).run(problem, seed=1)
    path = next(tmp_path.rglob("*.sqlite3"))
    for cls in [CalReader, InfReader, AnaReader]:
        assert cls.list_runs(tmp_path) == []
        with pytest.raises(ValueError, match="database"):
            cls(path)
    assert len(OptReader.list_runs(tmp_path)) == 1


# Regression source: test_remaining_review.py::testRunIdCollisionDoesNotChangePreviousRun
def testRunIdCollisionDoesNotChangePreviousRun(tmp_path, monkeypatch):
    import sqlite3
    from UQPyL.core import runtime
    from UQPyL.optimization.soea import GA
    from UQPyL.problem import Sphere

    problem = Sphere(nInput=2)
    problem.workDir = str(tmp_path)
    monkeypatch.setattr(runtime, "make_run_id", lambda *args: "fixed_run")
    method = GA(nPop=4, maxIters=0, saveFlag=True, logFlag=False, verboseFlag=False)
    method.run(problem, seed=1)
    with pytest.raises(sqlite3.IntegrityError):
        method.run(problem, seed=1)
    with closing(sqlite3.connect(next(tmp_path.rglob("*.sqlite3")))) as conn:
        assert conn.execute("SELECT status FROM run").fetchone()[0] == "finished"


# Regression source: test_remaining_review.py::testLogOnlyRunsHaveDistinctIdsAndFiles
def testLogOnlyRunsHaveDistinctIdsAndFiles(tmp_path):
    from UQPyL.optimization.soea import GA
    from UQPyL.problem import Sphere

    problem = Sphere(nInput=2)
    problem.workDir = str(tmp_path)
    method = GA(nPop=4, maxIters=0, logFlag=True, saveFlag=False, verboseFlag=False)
    ids = []
    for _ in range(2):
        method.run(problem, seed=1)
        ids.append(method.runId)
    assert len(set(ids)) == 2
    assert {p.stem for p in tmp_path.rglob("*.log")} == set(ids)
