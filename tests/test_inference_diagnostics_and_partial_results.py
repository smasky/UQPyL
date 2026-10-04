"""Inference diagnostics and partial results.

Migrated from test_review_remaining.py; original regression provenance is retained below.
"""

from copy import deepcopy
import inspect
import numpy as np
import pytest
from UQPyL.inference import MH, AMH, DEMC, DREAM_ZS, MH_Gibbs, InfReader
from UQPyL.inference.diagnostics import computeChainDiagnostics
from UQPyL.problem import Problem, ModelProblem


# Regression source: test_review_remaining.py::testPartialResultPreservesSparseSavedEndpoints
@pytest.mark.parametrize("failAt,saveFreq", [(1, 2), (7, 2), (7, 10), (None, 2)])
def testPartialResultPreservesSparseSavedEndpoints(failAt, saveFreq, tmp_path):
    calls = 0

    def objective(x):
        nonlocal calls
        calls += 1
        if calls == failAt:
            raise RuntimeError("simulator unavailable")
        return x[:, :1] ** 2

    problem = Problem(nInput=1, nObj=1, lb=-1, ub=1, objFunc=objective, optType="max")
    problem.workDir = str(tmp_path)
    method = MH(nChains=1, warmUp=0, maxIters=10, saveFreq=saveFreq, verboseFlag=False, saveFlag=True)
    if failAt is None:
        method.run(problem, seed=3)
    else:
        with pytest.raises(RuntimeError, match="simulator unavailable"):
            method.run(problem, seed=3)
    with InfReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        partial = reader.load_partial_result()
        assert partial["complete"] is False and partial["resumable"] is False
        assert partial["status"] == ("finished" if failAt is None else "failed")
        assert partial["sample_scope"] == "saved_chain_endpoints"
        assert partial["snapshot_count"] == len(reader.list_snapshots())
        if failAt == 1:
            assert partial["snapshots"] == [] and partial["last_saved_iter"] is None
        else:
            for snapshot in partial["snapshots"]:
                assert len(snapshot["members"]) == 1
                member = snapshot["members"][0]
                np.testing.assert_allclose(member["objs"], member["decs"] ** 2)
                assert member["log_prob"] == pytest.approx(member["objs"][0])
                assert "logProb" not in member
            if failAt is not None:
                assert [s["iter"] for s in partial["snapshots"]] == ([0, 2, 4] if saveFreq == 2 else [0])
            saved = deepcopy(partial)
            partial["snapshots"][0]["members"][0]["decs"][:] = 99
            np.testing.assert_array_equal(
                reader.load_partial_result()["snapshots"][0]["members"][0]["decs"],
                saved["snapshots"][0]["members"][0]["decs"],
            )
        if failAt is not None:
            with pytest.raises(ValueError, match="No result artifact"):
                reader.load_result()


# Regression source: test_review_remaining.py::testSplitRhatMatchesHandCalculatedExampleAndScaling
@pytest.mark.parametrize("factor", [1.0, 1e-200, 1e200, -1e200])
def testSplitRhatMatchesHandCalculatedExampleAndScaling(factor):
    # Halves: [0,2], [2,4], [4,6], [6,8]; W=2, var(means)=20/3.
    samples = np.array([[[0.0], [2.0], [4.0], [6.0]], [[2.0], [4.0], [6.0], [8.0]]]) * factor
    report = computeChainDiagnostics(samples)
    assert report["split_rhat"]["status"] == ["available"]
    assert report["split_rhat"]["values"] == pytest.approx([np.sqrt(0.5 + 10 / 3)])
    assert report["ess_bulk"]["status"] == ["available"]


# Regression source: test_review_remaining.py::testUnavailableDiagnosticsDoNotPretendConvergence
@pytest.mark.parametrize(
    "samples,status",
    [
        (np.zeros((1, 10, 1)), "insufficient_chains"),
        (np.zeros((2, 3, 1)), "insufficient_draws"),
        (np.zeros((2, 10, 1)), "constant_chain"),
        (np.full((2, 10, 1), np.nan), "nonfinite"),
        (np.full((2, 10, 1), "red"), "non_numeric"),
    ],
)
def testUnavailableDiagnosticsDoNotPretendConvergence(samples, status):
    metric = computeChainDiagnostics(samples)["split_rhat"]
    assert metric["status"] == [status] and metric["values"] == [None]


# Regression source: test_review_remaining.py::testOddDrawDiagnosticsAndIndependentVersusShiftedChains
def testOddDrawDiagnosticsAndIndependentVersusShiftedChains():
    samples = np.random.default_rng(5).normal(size=(4, 1001, 2))
    report = computeChainDiagnostics(samples)
    assert np.all(np.abs(np.array(report["split_rhat"]["values"]) - 1) < 0.01)
    changed = samples.copy()
    changed[:, 500] = 1e30
    changedReport = computeChainDiagnostics(changed)
    for key in ("split_rhat", "rhat", "ess_bulk"):
        assert changedReport[key] == report[key]  # Middle draw is omitted from split statistics.
    samples[0, :, 0] += 10
    assert computeChainDiagnostics(samples)["split_rhat"]["values"][0] > 2


# Regression source: test_review_remaining.py::testDiagnosticsAreExplicitAndResultExportsUseSnakeCase
def testDiagnosticsAreExplicitAndResultExportsUseSnakeCase():
    problem = Problem(nInput=1, nObj=1, lb=-2, ub=2, objFunc=lambda x: x**2)
    method = MH(nChains=2, warmUp=0, maxIters=20, verboseFlag=False)
    result = method.run(problem, seed=1)
    saved = result.decs.copy()
    assert result.diagnostics["chains"]["status"] == "not_computed"
    report = result.computeDiagnostics()
    assert method.state.diagnostics["chains"]["status"] == "not_computed"
    report["split_rhat"]["values"][:] = [99]
    assert result.diagnostics["chains"]["split_rhat"]["values"] != [99]
    np.testing.assert_array_equal(result.decs, saved)
    payload = result.toDict()
    assert all(key == key.lower() for key in payload)
    np.testing.assert_array_equal(payload["log_prob"], result.logProb)
    payload["decs"][:] = 99
    np.testing.assert_array_equal(result.decs, saved)


# Regression source: test_review_remaining.py::testAllInferenceConstructorsUseMaxIters
@pytest.mark.parametrize("methodClass", [MH, AMH, DEMC, DREAM_ZS, MH_Gibbs])
def testAllInferenceConstructorsUseMaxIters(methodClass):
    assert "maxIters" in inspect.signature(methodClass).parameters
    method = methodClass(nChains=4, maxIters=3, warmUp=0, verboseFlag=False)
    problem = Problem(nInput=1, nObj=1, lb=-1, ub=1, objFunc=lambda x: x**2)
    assert method.run(problem, seed=1).decs.shape == (4, 3, 1)
