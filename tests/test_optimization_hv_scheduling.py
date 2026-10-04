"""Optimization hv scheduling.

Migrated from test_review_c13_c14.py; original regression provenance is retained below.
"""

from copy import deepcopy
import importlib
import numpy as np
import pytest
from UQPyL.optimization import Population, OptReader
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.optimization.expensive import MOASMO
from UQPyL.optimization.metric import HV
from UQPyL.problem import Problem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


# Regression source: test_review_c13_c14.py::testHvConfigurationIsAvailableAndExportable
@pytest.mark.parametrize("methodClass", [NSGAII, NSGAIII, MOEAD, RVEA, MOASMO])
def testHvConfigurationIsAvailableAndExportable(methodClass):
    method = methodClass(hvFlag=False, hvFreq=7, hvSamples=321, **QUIET)
    assert method.get("hvFlag", "hvFreq", "hvSamples") == (False, 7, 321)
    assert method.exportConfig()["hvSamples"] == 321
    method.set("hvFlag", True)
    method.set("hvFreq", 2)
    method.set("hvSamples", 654)
    assert method.get("hvFlag", "hvFreq", "hvSamples") == (True, 2, 654)


# Regression source: test_review_c13_c14.py::testInvalidHvConfigurationFailsWithoutChangingSettings
@pytest.mark.parametrize(
    "key,value",
    [
        ("hvFlag", 0),
        ("hvFlag", None),
        ("hvFreq", 0),
        ("hvFreq", -1),
        ("hvFreq", True),
        ("hvFreq", 1.5),
        ("hvFreq", None),
        ("hvSamples", 0),
        ("hvSamples", -1),
        ("hvSamples", True),
        ("hvSamples", 2.5),
    ],
)
def testInvalidHvConfigurationFailsWithoutChangingSettings(key, value):
    with pytest.raises(ValueError, match=key):
        NSGAII(**{key: value}, **QUIET)
    method = NSGAII(**QUIET)
    expected = method.get(key)
    with pytest.raises(ValueError, match=key):
        method.set(key, value)
    assert method.get(key) == expected


# Regression source: test_review_c13_c14.py::testHvScheduleFinalResultAndSqliteAlignment
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("lastIter", [0, 3, 5])
def testHvScheduleFinalResultAndSqliteAlignment(enabled, lastIter, tmp_path, monkeypatch):
    module = importlib.import_module("UQPyL.optimization.runtime.result")
    original = module.HV
    calls = []

    def hv(*args, **kwargs):
        calls.append(method.iters)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "HV", hv)
    problem = Problem(nInput=1, nObj=2, lb=0, ub=1, objFunc=lambda x: np.column_stack([x[:, 0]] * 2))
    problem.workDir = str(tmp_path)
    method = NSGAII(
        hvFlag=enabled,
        hvFreq=3,
        hvSamples=321,
        hvRefPoint=[2, 2],
        saveFlag=True,
        saveFreq=1,
        verboseFlag=False,
        logFlag=False,
    )
    method.setup(problem, seed=3)
    initialRng = deepcopy(method.rng.bit_generator.state)
    for iteration in range(lastIter + 1):
        method.FEs += 1
        value = 0.9 - 0.1 * iteration
        method.update(Population([[0.5]], [[value, value]]), completed=iteration > 0)
        expected = (2 - value) ** 2 if enabled and iteration % 3 == 0 else None
        if expected is None:
            assert method.state.bestMetric is None
        else:
            assert method.state.bestMetric == pytest.approx(expected)
    result = method.finalize()
    expectedIndices = sorted({0, *range(3, lastIter + 1, 3), lastIter}) if enabled else []
    assert calls == expectedIndices
    assert method.rng.bit_generator.state == initialRng
    for iteration, value in enumerate(result.history.metrics):
        if iteration in expectedIndices:
            assert value == pytest.approx((2 - (0.9 - 0.1 * iteration)) ** 2)
        else:
            assert value is None
    assert result.bestMetric == result.history.bestMetricHistory[-1] == result.history.metrics[-1]
    with OptReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
        assert loaded.history.metrics == result.history.metrics
        assert loaded.history.iterToFEs == result.history.iterToFEs
        assert loaded.extra["hv_enabled"] == enabled
        assert loaded.extra["hv_freq"] == 3 and loaded.extra["hv_samples"] == 321
        restored = reader.load_algorithm()
        assert restored.get("hvFlag", "hvFreq", "hvSamples") == (enabled, 3, 321)


# Regression source: test_review_c13_c14.py::testHvPolicyDoesNotChangeSearchTrajectory
@pytest.mark.parametrize("methodClass", [NSGAII, NSGAIII, MOEAD, RVEA])
@pytest.mark.parametrize("nObj", [2, 4])
def testHvPolicyDoesNotChangeSearchTrajectory(methodClass, nObj):
    problem = Problem(
        nInput=2,
        nObj=nObj,
        lb=0,
        ub=1,
        objFunc=lambda x: np.column_stack([x[:, 0], 1 - x[:, 0], x[:, 1], 1 - x[:, 1]])[:, :nObj],
    )
    options = dict(nPop=12, maxIters=3, historyFreq=1, **QUIET)
    results = [
        methodClass(**policy, **options).run(problem, seed=2)
        for policy in (dict(hvFlag=False), dict(hvFreq=1, hvSamples=500), dict(hvFreq=10, hvSamples=2000))
    ]
    for result in results[1:]:
        np.testing.assert_array_equal(result.bestDecs, results[0].bestDecs)
        np.testing.assert_array_equal(result.bestObjs, results[0].bestObjs)
        assert result.FEs == results[0].FEs
        assert result.history.improvedHistory == results[0].history.improvedHistory
        for expected, actual in zip(results[0].history.populations, result.history.populations):
            np.testing.assert_array_equal(expected["decs"], actual["decs"])
            np.testing.assert_array_equal(expected["objs"], actual["objs"])
        expectedHv = HV(
            result.bestObjs,
            refPoint=result.extra["hv_reference_point"],
            normalize=False,
            nSamples=result.extra["hv_samples"],
            rng=np.random.default_rng(0),
        )
        assert result.bestMetric == expectedHv
    assert all(value is None for value in results[0].history.metrics)


# Regression source: test_review_c13_c14.py::testUnchangedFrontReusesHvAtScheduledIterations
def testUnchangedFrontReusesHvAtScheduledIterations(monkeypatch):
    module = importlib.import_module("UQPyL.optimization.runtime.result")
    original = module.HV
    calls = []

    def hv(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "HV", hv)
    problem = Problem(nInput=1, nObj=2, lb=0, ub=1, objFunc=lambda x: np.ones((len(x), 2)))
    result = NSGAII(nPop=4, maxIters=5, hvFreq=2, **QUIET).run(problem, seed=3)
    assert len(calls) == 1
    assert [i for i, value in enumerate(result.history.metrics) if value is not None] == [0, 2, 4, 5]


# Regression source: test_review_c13_c14.py::testAutomaticReferenceDoesNotDependOnHvSchedule
def testAutomaticReferenceDoesNotDependOnHvSchedule():
    problem = Problem(
        nInput=1, nObj=2, nCon=1, lb=0, ub=1, objFunc=lambda x: np.ones((len(x), 2)), conFunc=lambda x: x - 0.5
    )
    references = []
    for frequency in [1, 10]:
        method = NSGAII(hvFreq=frequency, **QUIET)
        method.setup(problem, seed=1)
        method.update(Population([[0.8]], [[1.0, 1.0]], [[1.0]]))
        method.update(Population([[0.4]], [[2.0, 2.0]], [[0.0]]), completed=True)
        method.update(Population([[0.3]], [[1.0, 1.0]], [[0.0]]), completed=True)
        references.append(method.finalize().extra["hv_reference_point"])
    np.testing.assert_array_equal(references[0], references[1])


# Regression source: test_review_c13_c14.py::testMoasmoDefaultInnerSearchSkipsUnusedHv
def testMoasmoDefaultInnerSearchSkipsUnusedHv():
    assert MOASMO(**QUIET).optimizer.get("hvFlag") is False
