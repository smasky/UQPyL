"""Inference result isolation and directions.

Migrated from test_review_c01_c05.py; original regression provenance is retained below.
"""

import pickle
import numpy as np
import pytest
from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS, InfReader
from UQPyL.problem import Problem, ModelProblem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)

METHODS = [MH, MH_Gibbs, AMH, DEMC, DREAM_ZS]


def makeInference(methodClass, **kwargs):
    key = "maxIters"
    options = dict(QUIET)
    options.update(kwargs)
    return methodClass(nChains=4, warmUp=1, **{key: 8}, **options)


# Regression source: test_review_c01_c05.py::testInferencePreviousResultSurvivesReuseResetAndFailure
@pytest.mark.parametrize("methodClass", METHODS)
def testInferencePreviousResultSurvivesReuseResetAndFailure(methodClass):
    problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=lambda x: np.sum(x * x, axis=1)[:, None])
    method = makeInference(methodClass)
    first = method.run(problem, seed=2)
    saved = pickle.dumps(first)
    method.maxIters = 3
    second = method.run(problem, seed=3)
    assert first.history is not second.history
    assert pickle.dumps(first) == saved
    method.reset()
    assert pickle.dumps(first) == saved
    with pytest.raises(ValueError):
        method.run(Problem(nInput=2, nObj=2, lb=0.0, ub=1.0, objFunc=lambda x: x), seed=2)
    assert pickle.dumps(first) == saved


# Regression source: test_review_c01_c05.py::testInferenceNestedStateIsolation
@pytest.mark.parametrize("field", ["settings", "history", "diagnostics", "extra"])
@pytest.mark.parametrize("changeResult", [True, False])
def testInferenceNestedStateIsolation(field, changeResult):
    problem = Problem(nInput=1, nObj=1, lb=0.0, ub=1.0, objFunc=lambda x: x)
    method = makeInference(MH)
    method.run(problem, seed=2)
    original = np.arange(8.0).reshape(2, 4)
    nested = {"rows": [original[:, ::2]], "labels": ["initial"]}
    if field == "settings":
        method.set("nested", nested)
        source = method.params.asDict()
    elif field == "history":
        method.state.history.snapshots.append(nested)
        source = method.state.history
    else:
        source = getattr(method.state, field)
        source["nested"] = nested
    first, second = method.buildResult(), method.buildResult()
    expectedFirst, expectedSecond, expectedSource = map(pickle.dumps, (first, second, source))
    target = getattr(first, field) if changeResult else source
    entry = target.snapshots[-1] if field == "history" else target["nested"]
    entry["rows"][0][:] = -99
    entry["labels"].append("changed")
    assert pickle.dumps(second) == expectedSecond
    assert pickle.dumps(source if changeResult else first) == (expectedSource if changeResult else expectedFirst)


# Regression source: test_review_c01_c05.py::testInferencePublicObjectiveDirectionsAndSamplingUnchanged
@pytest.mark.parametrize("methodClass", METHODS)
@pytest.mark.parametrize("direction", ["min", "max"])
def testInferencePublicObjectiveDirectionsAndSamplingUnchanged(methodClass, direction):
    objective = lambda x: x[:, :1] + 2 * x[:, 1:2]
    problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objective, optType=direction)
    result = makeInference(methodClass).run(problem, seed=4)
    values = objective(result.decs.reshape(-1, 2)).reshape(result.objs.shape)
    np.testing.assert_allclose(result.objs, values)
    np.testing.assert_allclose(result.bestObjs, objective(result.bestDecs))
    np.testing.assert_allclose(result.logProb, (-values * problem.opt)[..., 0])
    # An equivalent minimization energy must produce the very same chain.
    energy = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=lambda x: objective(x) * problem.opt)
    control = makeInference(methodClass).run(energy, seed=4)
    np.testing.assert_array_equal(result.decs, control.decs)
    np.testing.assert_array_equal(result.accepted, control.accepted)
    np.testing.assert_array_equal(result.logProb, control.logProb)


# Regression source: test_review_c01_c05.py::testInferenceSqliteSnapshotsUseOriginalObjectives
@pytest.mark.parametrize("direction", ["min", "max"])
def testInferenceSqliteSnapshotsUseOriginalObjectives(tmp_path, direction):
    problem = Problem(nInput=1, nObj=1, lb=0.0, ub=1.0, objFunc=lambda x: x + 1, optType=direction)
    problem.workDir = str(tmp_path)
    result = makeInference(MH, saveFlag=True, saveFreq=2).run(problem, seed=3)
    with InfReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
        np.testing.assert_array_equal(loaded.objs, result.objs)
        for snapshot in reader.list_snapshots():
            for member in reader.load_snapshot_members(snapshot["snapshotId"]):
                np.testing.assert_allclose(member["objs"], (np.asarray(member["decs"]) + 1))
                assert member["logProb"] == pytest.approx(float(-member["objs"][0] * problem.opt))
