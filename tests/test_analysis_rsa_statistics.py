"""Analysis rsa statistics.

Migrated from test_review_a01_a06.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from scipy.stats import cramervonmises_2samp
from UQPyL.problem import Problem
from UQPyL.analysis import RSA
from UQPyL.analysis.runtime import AnaReader


# Regression source: test_review_a01_a06.py::testRsaBinaryOutputMatchesIndependentTwoSampleStatistic
def testRsaBinaryOutputMatchesIndependentTwoSampleStatistic():
    x = np.linspace(0, 1, 40)[:, None]
    y = (x > 0.5).astype(float)
    p = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: (x > 0.5).astype(float))
    result = RSA(verboseFlag=False).analyze(p, x, y)
    expected = cramervonmises_2samp(x[:20, 0], x[20:, 0]).statistic
    assert result["S1"].values[0, 0] == pytest.approx(expected)
    assert result["S1_norm"].values[0, 0] == 1


# Regression source: test_review_a01_a06.py::testRsaConstantOutputRemainsFiniteZero
def testRsaConstantOutputRemainsFiniteZero():
    x = np.linspace(0, 1, 8)[:, None]
    p = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: np.ones_like(x))
    result = RSA(verboseFlag=False).analyze(p, x, np.ones_like(x))
    np.testing.assert_array_equal(result["S1"].values, [[0.0]])
    assert result.extra["rsa_regions"]["outputs"][0]["status"] == "constant_output"


@pytest.mark.parametrize("scale", [1e-200, 1e308, -1e308])
def testRsaOutputUnitsPreserveHandDerivedRankStatistic(scale):
    x = np.linspace(0, 1, 4)[:, None]
    y = np.array([-1, -1, 1, 1], dtype=float)[:, None] * scale
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    originalY = y.copy()
    result = RSA(nRegion=2, verboseFlag=False).analyze(problem, x, y)
    # Pooled ranks 1,2 versus 3,4: T = 1 - 15/24 = 3/8.
    np.testing.assert_allclose(result["S1"].values, [[3 / 8]])
    np.testing.assert_array_equal(result["S1_norm"].values, [[1.0]])
    np.testing.assert_array_equal(y, originalY)
    np.testing.assert_array_equal(result.Y, originalY)


def testRsaPreservesRegionsAcrossExtremelyUnequalOutputMagnitudes():
    x = np.linspace(0, 1, 4)[:, None]
    y = np.array([-1e308, -1e-200, 1e-200, 1e308])[:, None]
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    result = RSA(nRegion=2, verboseFlag=False).analyze(problem, x, y)
    np.testing.assert_allclose(result["S1"].values, [[3 / 8]])


@pytest.mark.parametrize("arrayName", ["X", "Y"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def testRsaRejectsNonfiniteDataInsteadOfReturningZero(arrayName, invalid):
    x = np.linspace(0, 1, 4)[:, None]
    y = np.array([-1, -1, 1, 1], dtype=float)[:, None]
    (x if arrayName == "X" else y)[0, 0] = invalid
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    with pytest.raises(ValueError, match="finite"):
        RSA(nRegion=2, verboseFlag=False).analyze(problem, x, y)


@pytest.mark.parametrize("nSamples,nRegion", [(3, 2), (20, 20)])
def testRsaWarnsAndMarksUnavailableWhenEveryRegionIsTooSmall(nSamples, nRegion):
    x = np.linspace(0, 1, nSamples)[:, None]
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    with pytest.warns(RuntimeWarning, match="no valid region comparisons") as caught:
        result = RSA(nRegion=nRegion, verboseFlag=False).analyze(problem, x, x)
    assert len(caught) == 1
    assert "zero placeholders" in str(caught[0].message)
    np.testing.assert_array_equal(result["S1"].values, [[0.0]])
    np.testing.assert_array_equal(result["S1_norm"].values, [[0.0]])
    diagnostics = result.extra["rsa_regions"]
    assert diagnostics["n_samples"] == nSamples
    assert diagnostics["n_regions"] == nRegion
    output = diagnostics["outputs"][0]
    assert output["status"] == "insufficient_samples"
    assert output["valid_region_count"] == 0
    assert sum(output["region_sample_counts"]) == nSamples


def testRsaWarnsForRareBinaryOutputWithoutEnoughComplementSamples():
    x = np.linspace(0, 1, 20)[:, None]
    y = np.zeros_like(x)
    y[-1] = 1
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    with pytest.warns(RuntimeWarning, match="at least two samples"):
        result = RSA(verboseFlag=False).analyze(problem, x, y)
    output = result.extra["rsa_regions"]["outputs"][0]
    assert output["status"] == "insufficient_samples"
    assert output["valid_region_count"] == 0
    assert sorted(output["region_sample_counts"])[-2:] == [1, 19]


def testRsaAdequateRegionsKeepHandDerivedStatisticAndResetDiagnostics():
    x = np.linspace(0, 1, 20)[:, None]
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    method = RSA(verboseFlag=False)
    with pytest.warns(RuntimeWarning, match="no valid region comparisons"):
        unavailable = method.analyze(problem, x)
    method.set("nRegion", np.int64(2))
    result = method.analyze(problem, x)
    # Pooled ranks 1..10 vs 11..20: U=10000, T=10000/2000-399/120.
    np.testing.assert_allclose(result["S1"].values, [[1.675]], atol=1e-14)
    np.testing.assert_array_equal(result["S1_norm"].values, [[1.0]])
    output = result.extra["rsa_regions"]["outputs"][0]
    assert output["status"] == "estimated"
    assert output["valid_region_count"] == 2
    assert output["region_sample_counts"] == [10, 10]
    assert unavailable.extra["rsa_regions"]["outputs"][0]["status"] == "insufficient_samples"


@pytest.mark.parametrize("provideY", [False, True])
def testRsaSelectedOutputsWarnOnlyForUnavailableColumn(provideY):
    x = np.linspace(0, 1, 20)[:, None]
    y = np.column_stack([x[:, 0], (x[:, 0] > 0.5).astype(float), np.ones(20)])
    calls = []

    def objective(samples):
        calls.append(len(samples))
        return y.copy()

    problem = Problem(nInput=1, nObj=3, lb=0, ub=1, objLabels=["active", "binary", "constant"], objFunc=objective)
    with pytest.warns(RuntimeWarning, match="output 'active'") as caught:
        result = RSA(verboseFlag=False).analyze(problem, x, y if provideY else None, index=[2, 0, 1])
    assert len(caught) == 1
    assert calls == ([] if provideY else [20])
    outputs = result.extra["rsa_regions"]["outputs"]
    assert [output["output_label"] for output in outputs] == ["constant", "active", "binary"]
    assert [output["status"] for output in outputs] == ["constant_output", "insufficient_samples", "estimated"]
    assert [output["valid_region_count"] for output in outputs] == [0, 0, 2]
    np.testing.assert_allclose(result["S1"].values[:, 0], [0, 0, 1.675], atol=1e-14)
    np.testing.assert_array_equal(result["S1_norm"].values[:, 0], [0, 0, 1])
    np.testing.assert_array_equal(result.Y, y[:, [2, 0, 1]])


def testRsaUnavailableStatusSurvivesSqlitePersistence(tmp_path):
    x = np.linspace(0, 1, 20)[:, None]
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    problem.workDir = str(tmp_path)
    with pytest.warns(RuntimeWarning, match="no valid region comparisons"):
        result = RSA(verboseFlag=False, saveFlag=True).analyze(problem, x)
    with AnaReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
        assert reader.get_run_summary()["status"] == "finished"
    assert loaded.extra["rsa_regions"] == result.extra["rsa_regions"]
    assert loaded.extra["rsa_regions"]["outputs"][0]["status"] == "insufficient_samples"
    np.testing.assert_array_equal(loaded["S1"].values, result["S1"].values)


@pytest.mark.parametrize("nRegion", [0, 1, -1, 2.5, True, np.bool_(False), "2", None])
def testRsaRejectsInvalidRegionCountBeforeModelEvaluation(nRegion):
    x = np.linspace(0, 1, 20)[:, None]
    calls = []

    def objective(samples):
        calls.append(len(samples))
        return samples

    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=objective)
    method = RSA(nRegion=2, verboseFlag=False)
    method.set("nRegion", nRegion)
    with pytest.raises(ValueError, match="nRegion.*integer.*at least 2"):
        method.analyze(problem, x)
    assert calls == []
