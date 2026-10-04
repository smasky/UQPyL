"""Optimization configuration lifecycle.

Migrated from test_review_c06_c12.py; original regression provenance is retained below.
"""

import pickle
import numpy as np
import pytest
from UQPyL.optimization.expensive import MOASMO
from UQPyL.optimization.moea import NSGAII
from UQPyL.optimization.soea import GA
from UQPyL.optimization.runtime import OptReader
from UQPyL.problem import Problem, ModelProblem
from UQPyL.surrogate import MultiSurrogate
from UQPyL.surrogate.rbf import RBF

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def makeProblem(nInput=2, nObj=2, calls=None):
    def objective(x):
        if calls is not None:
            calls.append(x.copy())
        return np.column_stack([np.sum((x - j / nObj) ** 2, axis=1) for j in range(nObj)])

    return Problem(nInput=nInput, nObj=nObj, lb=0.0, ub=1.0, objFunc=objective, name=f"problem_{nInput}_{nObj}")


def makeMoasmo(**kwargs):
    inner = NSGAII(nPop=8, maxFEs=16, maxIters=1, **QUIET)
    return MOASMO(nInit=8, pct=0.25, maxIters=1, optimizer=inner, **QUIET, **kwargs)


# Regression source: test_review_c06_c12.py::testMoasmoDefaultPopulationSizeIsUsed
@pytest.mark.parametrize("nPop", [8, 24, 100])
def testMoasmoDefaultPopulationSizeIsUsed(nPop):
    method = MOASMO(nPop=nPop, **QUIET)
    method.optimizer.set("maxIters", 0)
    result = method.optimizer.run(makeProblem(), seed=9)
    assert result.FEs == nPop
    assert len(result.history.populations[-1]["decs"]) == nPop


# Regression source: test_review_c06_c12.py::testMoasmoCustomOptimizerKeepsPopulationSize
def testMoasmoCustomOptimizerKeepsPopulationSize():
    inner = NSGAII(nPop=12, maxIters=0, **QUIET)
    method = MOASMO(nPop=24, optimizer=inner, **QUIET)
    assert method.optimizer is inner
    assert method.optimizer.run(makeProblem(), seed=9).FEs == 12


# Regression source: test_review_c06_c12.py::testMoasmoAutomaticSurrogatesFollowCurrentProblem
def testMoasmoAutomaticSurrogatesFollowCurrentProblem():
    method = makeMoasmo()
    previousModels = []
    for nInput, nObj in [(2, 2), (2, 3), (3, 3), (2, 2)]:
        problem = makeProblem(nInput, nObj)
        actual = method.run(problem, seed=11)
        expected = makeMoasmo().run(problem, seed=11)
        assert len(method.surrogates.models_list) == nObj
        assert not any(model is old for model in method.surrogates.models_list for old in previousModels)
        np.testing.assert_allclose(actual.bestDecs, expected.bestDecs)
        np.testing.assert_allclose(actual.bestObjs, expected.bestObjs)
        assert actual.FEs == expected.FEs == 10
        previousModels.extend(method.surrogates.models_list)


# Regression source: test_review_c06_c12.py::testMoasmoRetainsSuppliedSurrogatesAndChecksMismatchBeforeEvaluation
def testMoasmoRetainsSuppliedSurrogatesAndChecksMismatchBeforeEvaluation():
    models = [RBF(), RBF()]
    surrogate = MultiSurrogate(2, models_list=models)
    method = makeMoasmo(surrogates=surrogate)
    method.run(makeProblem(), seed=1)
    method.run(makeProblem(), seed=2)
    assert method.surrogates is surrogate
    assert all(actual is supplied for actual, supplied in zip(surrogate.models_list, models))
    calls = []
    with pytest.raises(ValueError, match="n_surrogates"):
        method.run(makeProblem(nObj=3, calls=calls), seed=1)
    assert calls == []


# Regression source: test_review_c06_c12.py::testMoasmoMalformedCustomSurrogatesFailBeforeEvaluation
@pytest.mark.parametrize("defect", ["empty", "duplicate", "invalid"])
def testMoasmoMalformedCustomSurrogatesFailBeforeEvaluation(defect):
    surrogate = MultiSurrogate(2, models_list=[RBF(), RBF()])
    if defect == "empty":
        surrogate.models_list.clear()
    elif defect == "duplicate":
        surrogate.models_list[1] = surrogate.models_list[0]
    else:
        surrogate = object()
    calls = []
    with pytest.raises((ValueError, TypeError)):
        makeMoasmo(surrogates=surrogate).run(makeProblem(calls=calls), seed=1)
    assert calls == []


# Regression source: test_review_c06_c12.py::testMoasmoAllowsExplicitReplacementOfAutomaticSurrogates
def testMoasmoAllowsExplicitReplacementOfAutomaticSurrogates():
    method = makeMoasmo()
    method.run(makeProblem(), seed=1)
    supplied = MultiSurrogate(2, models_list=[RBF(), RBF()])
    method.surrogates = supplied
    method.run(makeProblem(), seed=1)
    assert method.surrogates is supplied


# Regression source: test_review_c06_c12.py::testOptimizationSetterControlsActualTermination
@pytest.mark.parametrize("key,value", [("maxIters", 0), ("maxFEs", 0), ("maxTolerates", 0)])
def testOptimizationSetterControlsActualTermination(key, value):
    method = GA(nPop=8, maxIters=2, **QUIET)
    method.set(key, value)
    assert method.get(key) == method.exportConfig()[key] == value
    result = method.run(makeProblem(nObj=1), seed=1)
    assert result.iters == 0 and result.FEs == 8


# Regression source: test_review_c06_c12.py::testOptimizationCommonConfigurationUsesLiveAttributes
@pytest.mark.parametrize(
    "key,value",
    [
        ("maxIters", 2),
        ("maxFEs", 30),
        ("maxTolerates", 3),
        ("tolerate", None),
        ("verboseFlag", True),
        ("verboseFreq", 3),
        ("logFlag", True),
        ("saveFlag", True),
        ("saveFreq", 2),
        ("historyFreq", None),
    ],
)
def testOptimizationCommonConfigurationUsesLiveAttributes(key, value):
    method = GA(**QUIET)
    method.set(key, value)
    attr = "maxIter" if key == "maxIters" else key
    assert method.get(key) == getattr(method, attr) == method.exportConfig()[key] == value
    # Direct attribute configuration remains readable through the common API.
    setattr(method, attr, value if value is None else 1)
    assert method.get(key) == getattr(method, attr) == method.exportConfig()[key]
    assert method.get("maxIters", "nPop") == (method.maxIter, method.params.get("nPop"))


# Regression source: test_review_c06_c12.py::testOptimizationHistoryAndReferencePointConfiguration
def testOptimizationHistoryAndReferencePointConfiguration():
    method = NSGAII(nPop=8, maxIters=2, **QUIET)
    reference = np.array([5.0, 6.0])
    method.set("hvRefPoint", reference)
    reference[:] = 0
    read = method.get("hvRefPoint")
    read[:] = 0
    method.set("historyFreq", None)
    result = method.run(makeProblem(), seed=2)
    assert len(result.history.populations) == 1
    np.testing.assert_array_equal(result.extra["hv_reference_point"], [5, 6])
    assert method.exportConfig()["hvRefPoint"] == [5, 6]
    with pytest.raises(ValueError, match="historyFreq"):
        method.set("historyFreq", 0)
    assert method.get("historyFreq") is None


# Regression source: test_review_c06_c12.py::testOptimizationResultIdentitySurvivesReuseAndDatabaseRoundTrip
@pytest.mark.parametrize("save", [False, True])
def testOptimizationResultIdentitySurvivesReuseAndDatabaseRoundTrip(save, tmp_path):
    problem = makeProblem(nObj=1)
    problem.workDir = str(tmp_path)
    method = GA(nPop=8, maxIters=2, **QUIET)
    method.set("saveFlag", save)
    method.set("saveFreq", 1)
    result = method.run(problem, seed=3)
    summary = result.summary()
    expected = dict(
        run_id=method.runId,
        method="GA",
        problem_name=problem.name,
        n_input=2,
        n_output=1,
        n_con=0,
        created_at=method.state.createdAt,
    )
    assert all(summary[key] == value for key, value in expected.items())
    assert summary["run_id"] and summary["created_at"]
    assert all(result.toDict()[key] == value for key, value in expected.items())
    saved = pickle.dumps(result)
    if save:
        with OptReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
            loaded = reader.load_result()
            assert loaded.summary() == summary
            assert all(reader.get_run_summary()[key] == value for key, value in expected.items())
            assert reader.load_algorithm().get("saveFreq") == 1
    method.set("saveFlag", False)
    second = method.run(makeProblem(nInput=3, nObj=1), seed=3)
    assert second.runId != result.runId and second.nInput == 3
    method.reset()
    assert pickle.dumps(result) == saved
