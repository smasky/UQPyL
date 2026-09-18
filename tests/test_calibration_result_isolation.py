from copy import deepcopy
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from UQPyL.calibration import ES, GLUE, IES, SUFI2, CalReader
from UQPyL.problem import ModelProblem


def simulate(X):
    return np.column_stack((X[:, 0], X[:, 1] ** 2))[:, :, None]


def makeProblem():
    return ModelProblem(nInput=2, lb=0., ub=3., simFunc=simulate,
                        obs=np.array([[1.], [2.]]), seriesLabels=["Q"])


def runMethod(method, problem):
    samples = np.array([[0., .5], [2., 1.], [1.5, 2.], [1., 1.4]])
    if isinstance(method, SUFI2):
        return method.run(problem, eliteSize=3, seed=7)
    if isinstance(method, GLUE):
        return method.run(problem, samples, threshold=10.)
    return method.run(problem, samples)


def makeMethod(methodClass, **options):
    if methodClass is IES:
        return IES(maxIters=2, lam=1e-6, **options)
    if methodClass is SUFI2:
        return SUFI2(maxIters=2, nSamples=12, **options)
    return methodClass(**options)


def assertEqual(actual, expected):
    if is_dataclass(expected):
        for item in fields(expected):
            assertEqual(getattr(actual, item.name), getattr(expected, item.name))
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assertEqual(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assertEqual(left, right)
    else:
        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("methodClass", [ES, IES, GLUE, SUFI2])
@pytest.mark.parametrize("changeResult", [False, True])
def testRealDiagnosticArraysAreIsolatedInBothDirections(methodClass, changeResult):
    method = makeMethod(methodClass)
    result = runMethod(method, makeProblem())
    expected = deepcopy(result)
    stateScores = method.state.diagnostics["scores"].copy()
    if changeResult:
        result.diagnostics["scores"][:] = -999.
        np.testing.assert_array_equal(method.state.diagnostics["scores"], stateScores)
    else:
        method.state.diagnostics["scores"][:] = -999.
        assertEqual(result, expected)


@pytest.mark.parametrize("methodClass", [IES, SUFI2])
def testReusingAlgorithmPreservesEntirePreviousResult(methodClass):
    method = makeMethod(methodClass)
    problem = makeProblem()
    first = runMethod(method, problem)
    expected = deepcopy(first)
    assert len(first.history.metricsHistory) == 2
    method.set("maxIters", 1)
    second = runMethod(method, problem)
    assert len(second.history.metricsHistory) == 1
    assertEqual(first, expected)
    assert first.history is not second.history
    assert first.summary()["iters"] == 2


@pytest.mark.parametrize("fieldName", ["history", "diagnostics", "extra", "settings"])
@pytest.mark.parametrize("changeResult", [False, True])
def testNestedContainersAndArrayViewsAreIsolated(fieldName, changeResult):
    method = makeMethod(IES)
    runMethod(method, makeProblem())
    parent = np.arange(12., dtype=float).reshape(3, 4)
    nested = {"items": [{"values": parent[:, ::2], "labels": ["original"]}]}
    if fieldName == "history":
        method.state.history.metricsHistory.append(nested)
        source = method.state.history
    elif fieldName == "settings":
        method.set("nested", nested)
        source = method.params.asDict()
    else:
        getattr(method.state, fieldName)["nested"] = nested
        source = getattr(method.state, fieldName)
    first = method.state.buildResult()
    second = method.state.buildResult()
    expectedFirst, expectedSecond, expectedSource = deepcopy((first, second, source))
    target = getattr(first, fieldName) if changeResult else source
    entry = target.metricsHistory[-1] if fieldName == "history" else target["nested"]
    entry["items"][0]["values"][:] = -99.
    entry["items"][0]["labels"].append("changed")
    entry["items"].append({"new": True})
    if changeResult:
        assertEqual(source, expectedSource)
    else:
        assertEqual(first, expectedFirst)
    assertEqual(second, expectedSecond)


def testResetAndFailedNextRunPreservePreviousResult():
    method = makeMethod(IES)
    problem = makeProblem()
    first = runMethod(method, problem)
    expected = deepcopy(first)
    with pytest.raises(ValueError):
        method.run(problem, np.array([[np.nan, 1.]]))
    assertEqual(first, expected)
    method.state.reset()
    assertEqual(first, expected)


def testBuildingSnapshotsDoesNotEvaluateModelOrChangeValues():
    method = makeMethod(IES)
    problem = makeProblem()
    first = runMethod(method, problem)
    expected = deepcopy(first)

    def failOnSimulation(X):
        raise AssertionError("Building a result must not simulate the model")

    problem.simFunc = failOnSimulation
    assertEqual(method.state.buildResult(), expected)
    assertEqual(method.state.buildResult(), expected)


def testSqliteResultMatchesIndependentInMemorySnapshot(tmp_path):
    method = makeMethod(SUFI2, saveFlag=True)
    problem = makeProblem()
    problem.workDir = str(tmp_path)
    result = runMethod(method, problem)
    expected = deepcopy(result)
    method.state.reset()
    assertEqual(result, expected)
    reader = CalReader(next(tmp_path.rglob("*.sqlite3")))
    try:
        assertEqual(reader.load_result(), expected)
    finally:
        reader.close()
