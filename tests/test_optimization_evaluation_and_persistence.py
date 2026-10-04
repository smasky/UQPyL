"""Optimization evaluation and persistence.

Migrated from test_remaining_review.py; original regression provenance is retained below.
"""

import numpy as np
import pytest


# Regression source: test_remaining_review.py::testPreEvaluatedPopulationContract
@pytest.mark.parametrize("bad", ["missingCons", "rows", "columns", "valid"])
def testPreEvaluatedPopulationContract(bad):
    from UQPyL.problem import Problem
    from UQPyL.optimization import Population
    from UQPyL.optimization.soea import GA

    def unexpected(X):
        raise AssertionError("Pre-evaluated data must not be evaluated again")

    problem = Problem(nInput=1, nObj=1, nCon=1, lb=0.0, ub=1.0, objFunc=unexpected, conFunc=unexpected)
    pop = Population(
        [[0.2], [0.8]],
        np.ones((1 if bad == "rows" else 2, 2 if bad == "columns" else 1)),
        None if bad == "missingCons" else np.zeros((2, 1)),
    )
    method = GA(nPop=2, maxIters=0, verboseFlag=False, logFlag=False, saveFlag=False)
    if bad == "valid":
        assert method.run(problem, initialPop=pop).FEs == 0
    else:
        with pytest.raises(ValueError):
            method.run(problem, initialPop=pop)


# Regression source: test_remaining_review.py::testSavedOptimizationResultPreservesMissingMetrics
def testSavedOptimizationResultPreservesMissingMetrics(tmp_path):
    from UQPyL.optimization.moea import NSGAII
    from UQPyL.optimization.runtime import OptReader
    from UQPyL.problem import Problem
    from UQPyL.viz.optimization import _history_xy

    problem = Problem(
        nInput=1,
        nObj=2,
        nCon=1,
        lb=0.0,
        ub=1.0,
        objFunc=lambda X: np.column_stack([X, 1 - X]),
        conFunc=lambda X: np.ones_like(X),
    )
    problem.workDir = str(tmp_path)
    method = NSGAII(nPop=4, maxIters=2, saveFlag=True, saveFreq=1, verboseFlag=False, logFlag=False)
    original = method.run(problem, seed=1)
    reader = OptReader(next(tmp_path.rglob("*.sqlite3")))
    try:
        result = reader.load_result()
        assert not result.bestFeasible
        assert len(result.history.populations) == len(result.history.snapshotIterToFEs) == 3
        np.testing.assert_array_equal(result.history.populations[-1]["decs"], reader.load_last_population().decs)
        assert result.toDict()["best_feasible"] is False
        np.testing.assert_array_equal(result.candidateObjs, original.candidateObjs)
        rows = reader.list_snapshots()
        for row, value in zip(rows, [None, 2.0, 3.0, 3.0]):
            reader.conn.execute("UPDATE snapshot SET hypervolume=? WHERE snapshotId=?", (value, row["snapshotId"]))
        result = reader.load_result()
        x, y = _history_xy(result, "fe")
        np.testing.assert_array_equal(x, [rows[1]["fe"], rows[2]["fe"]])
        np.testing.assert_array_equal(y, [2.0, 3.0])
    finally:
        reader.close()


# Regression source: test_remaining_review.py::testOptimizationElapsedTimeIsSaved
@pytest.mark.parametrize("verbose", [False, True])
def testOptimizationElapsedTimeIsSaved(tmp_path, verbose):
    import time
    from UQPyL.problem import Problem
    from UQPyL.optimization.soea import GA
    from UQPyL.optimization.runtime import OptReader

    def objective(X):
        time.sleep(0.005)
        return np.sum(X**2, axis=1, keepdims=True)

    problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objective)
    problem.workDir = str(tmp_path)
    result = GA(nPop=4, maxIters=2, tolerate=None, verboseFlag=verbose, saveFlag=True, logFlag=False, saveFreq=1).run(
        problem, seed=1
    )
    assert result.runtime >= 0.015
    with OptReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        elapsed = [row["elapsed"] for row in reader.list_snapshots()]
        assert elapsed == sorted(elapsed) and elapsed[0] > 0
        assert elapsed[-1] == reader.get_run()["runtime"] == result.runtime


# Regression source: test_remaining_review.py::testEveryOptimizationConfigurationCanBeRestored
@pytest.mark.parametrize("methodIndex", range(14))
def testEveryOptimizationConfigurationCanBeRestored(tmp_path, methodIndex):
    from optimization_test_support import METHODS, MULTI, makeMethod
    from UQPyL.problem import Problem
    from UQPyL.optimization.runtime import OptReader
    import warnings

    cls = METHODS[methodIndex]
    nObj = 2 if cls in MULTI else 1
    problem = Problem(
        nInput=2, nObj=nObj, lb=0.0, ub=1.0, objFunc=lambda X: np.column_stack([np.sum(X**2, axis=1)] * nObj)
    )
    problem.workDir = str(tmp_path)
    original = makeMethod(cls, 0)
    original.saveFlag = True
    original.run(problem, seed=2)
    with OptReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            restored = reader.load_algorithm()
        assert type(restored) is cls
        for name in ["maxFEs", "maxIter", "maxTolerates", "tolerate", "verboseFlag", "saveFlag", "historyFreq"]:
            assert getattr(restored, name) == getattr(original, name)
        if hasattr(original, "optimizer"):
            assert any("manually" in str(message.message) for message in messages)


# Regression source: test_remaining_review.py::testOptimizationConfigurationRoundTrip
def testOptimizationConfigurationRoundTrip(tmp_path):
    from UQPyL.optimization.soea import GA
    from UQPyL.optimization.runtime import OptReader
    from UQPyL.problem import Sphere

    problem = Sphere(nInput=2)
    problem.workDir = str(tmp_path)
    original = GA(
        nPop=4,
        maxFEs=18,
        maxIters=0,
        tolerate=None,
        maxTolerates=3,
        verboseFlag=False,
        logFlag=False,
        saveFlag=True,
        saveFreq=2,
        historyFreq=3,
    )
    original.run(problem, seed=1)
    reader = OptReader(next(tmp_path.rglob("*.sqlite3")))
    try:
        restored = reader.load_algorithm()
        for key in [
            "maxFEs",
            "maxIter",
            "tolerate",
            "maxTolerates",
            "verboseFlag",
            "logFlag",
            "saveFlag",
            "saveFreq",
            "historyFreq",
        ]:
            assert getattr(restored, key) == getattr(original, key)
        assert reader._parseValue("1+2") == "1+2"
    finally:
        reader.close()
