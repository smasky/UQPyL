"""Structured sensitivity designs and estimator denominators must be valid."""

import numpy as np
import pytest

from UQPyL.analysis import FAST, Sobol
from UQPyL.analysis.runtime import AnaReader
from UQPyL.doe import FASTDesign, SaltelliDesign
from UQPyL.problem import Problem


@pytest.mark.parametrize("rowChange", [-2, -1, 1])
def testFastRejectsMissingOrExtraRows(rowChange):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    x, meta = FASTDesign(M=4).sampleWithMeta(problem, 129, seed=3)
    y = x[:, 1:2]
    if rowChange < 0:
        x, y = x[:rowChange], y[:rowChange]
    else:
        x, y = np.vstack([x, x[-1:]]), np.vstack([y, [[1e200]]])
    with pytest.raises(ValueError, match="FAST.*rows"):
        FAST(verboseFlag=False).analyze(problem, x, y, meta)


@pytest.mark.parametrize("key,value", [("N", 128), ("blockSize", 128), ("M", 0), ("M", 2.5), ("N", True)])
def testFastRejectsInvalidMetadataBeforeEvaluation(key, value):
    calls = []

    def evaluate(x):
        calls.append(len(x))
        return x[:, 1:2]

    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=evaluate)
    x, meta = FASTDesign(M=4).sampleWithMeta(problem, 129, seed=3)
    meta[key] = value
    with pytest.raises(ValueError, match="FAST"):
        FAST(verboseFlag=False).analyze(problem, x, meta=meta)
    assert not calls


def testFastRequiresBaseSizeInMetadata():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    x, meta = FASTDesign().sampleWithMeta(problem, 129, seed=3)
    del meta["N"]
    with pytest.raises(ValueError, match="FAST.*N"):
        FAST(verboseFlag=False).analyze(problem, x, x[:, 1:2], meta)


def testFastRejectsInsufficientSpectralBudget():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    x = np.random.default_rng(17).random((128, 2))
    meta = dict(designType="fast", N=64, M=4, blockSize=64)
    with pytest.raises(ValueError, match=r"N > 4\*M\^2"):
        FAST(verboseFlag=False).analyze(problem, x, x[:, 1:2], meta)


def testFastCompleteDesignMatchesAnalyticalEffects():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    x, meta = FASTDesign(M=4).sampleWithMeta(problem, 129, seed=3)
    result = FAST(verboseFlag=False).analyze(problem, x, meta=meta)
    np.testing.assert_allclose(result["S1"].values, [[0, 1]], atol=0.003, rtol=0)
    np.testing.assert_allclose(result["ST"].values, [[0, 1]], atol=0.001, rtol=0)


@pytest.mark.parametrize("secondOrder", [False, True])
def testSobolDiagnosesZeroBaseVarianceWithVaryingHybrids(secondOrder):
    problem = Problem(
        nInput=2,
        nObj=1,
        lb=0,
        ub=1,
        objFunc=lambda x: ((x[:, 0] > 0.85) & (x[:, 1] > 0.85)).astype(float)[:, None],
    )
    x, meta = SaltelliDesign(secondOrder=secondOrder).sampleWithMeta(problem, 16, seed=1)
    y = problem.evaluate(x).objs
    step = 6 if secondOrder else 4
    assert np.var(np.r_[y[::step], y[step - 1 :: step]]) == 0
    assert np.var(y) > 0
    with pytest.raises(ValueError, match="base A/B.*increase"):
        Sobol(verboseFlag=False).analyze(problem, x, y, meta)


@pytest.mark.parametrize("secondOrder", [False, True])
def testSobolAdequateRareEventSamplesMatchPopulationReference(secondOrder):
    problem = Problem(
        nInput=2,
        nObj=1,
        lb=0,
        ub=1,
        objFunc=lambda x: ((x[:, 0] > 0.85) & (x[:, 1] > 0.85)).astype(float)[:, None],
    )
    x, meta = SaltelliDesign(secondOrder=secondOrder).sampleWithMeta(problem, 4096, seed=17)
    result = Sobol(verboseFlag=False).analyze(problem, x, meta=meta)
    # Independent conjunction of two Bernoulli(q) events.
    q = 0.15
    np.testing.assert_allclose(result["S1"].values, [[q / (1 + q)] * 2], atol=0.02, rtol=0)
    np.testing.assert_allclose(result["ST"].values, [[1 / (1 + q)] * 2], atol=0.02, rtol=0)
    if secondOrder:
        np.testing.assert_allclose(result["S2"].values, [[(1 - q) / (1 + q)]], atol=0.04, rtol=0)


@pytest.mark.parametrize("secondOrder", [False, True])
@pytest.mark.parametrize("rowChange", ["partialMissing", "blockMissing", "partialExtra", "blockExtra"])
def testSobolWarnsForChangedRowsAndOnlyUsesCompleteDesigns(secondOrder, rowChange):
    calls = []

    def objective(samples):
        calls.append(len(samples))
        return samples[:, :1] + 2 * samples[:, 1:2]

    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=objective)
    x, meta = SaltelliDesign(secondOrder=secondOrder).sampleWithMeta(problem, 128, seed=17)
    count = meta["blockSize"] if rowChange.startswith("block") else 1
    x = x[:-count] if rowChange.endswith("Missing") else np.vstack([x, x[:count]])
    with pytest.warns(RuntimeWarning, match="Sobol.*rows") as caught:
        result = Sobol(verboseFlag=False).analyze(problem, x, meta=meta)
    assert len(caught) == 1
    diagnostic = result.extra["sobol_design"]
    if rowChange.startswith("block"):
        assert calls == [len(x)]
        assert diagnostic["status"] == "recovered"
        assert diagnostic["effective_n"] == (127 if rowChange.endswith("Missing") else 129)
        assert diagnostic["effective_second_order"] == secondOrder
        np.testing.assert_allclose(result["S1"].values, [[0.2, 0.8]], atol=0.03, rtol=0)
    else:
        assert calls == []
        assert result.Y is None
        assert diagnostic["status"] == "not_estimated"
        assert diagnostic["metrics_available"] is False
        for metric in result.metrics:
            np.testing.assert_array_equal(metric.values, 0)
    assert result.meta["N"] == 128
    np.testing.assert_array_equal(result.X, x)


@pytest.mark.parametrize("key,value", [("N", 1), ("blockSize", 4), ("secondOrder", False)])
def testSobolWarnsAndRecoversContradictoryMetadataFromSampleStructure(key, value):
    calls = []

    def objective(samples):
        calls.append(len(samples))
        return samples[:, :1] + 2 * samples[:, 1:2]

    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=objective)
    x, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 128, seed=17)
    meta[key] = value
    originalMeta = dict(meta)
    # All 768 rows are also divisible by the wrong first-order block length 4.
    assert len(x) % (problem.nInput + 2) == 0
    with pytest.warns(RuntimeWarning, match="Sobol") as caught:
        result = Sobol(verboseFlag=False).analyze(problem, x, meta=meta)
    assert len(caught) == 1
    assert calls == [len(x)]
    diagnostic = result.extra["sobol_design"]
    assert diagnostic["status"] == "recovered"
    assert diagnostic["effective_n"] == 128
    assert diagnostic["effective_block_size"] == 6
    assert diagnostic["effective_second_order"] is True
    assert diagnostic["recovery_basis"] == "sample_structure"
    np.testing.assert_allclose(result["S1"].values, [[0.2, 0.8]], atol=0.03, rtol=0)
    assert result.meta == originalMeta == meta


@pytest.mark.parametrize(
    "key,value",
    [
        ("N", 0),
        ("N", True),
        ("N", 2.5),
        ("blockSize", True),
        ("blockSize", 6.0),
        ("secondOrder", 0),
        ("secondOrder", "False"),
    ],
)
def testSobolWarnsForInvalidMetadataTypesAndRecoversCompleteBlocks(key, value):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 8, seed=17)
    meta[key] = value
    with pytest.warns(RuntimeWarning, match=f"Sobol.*{key}"):
        result = Sobol(verboseFlag=False).analyze(problem, x, x[:, :1], meta)
    diagnostic = result.extra["sobol_design"]
    assert diagnostic["status"] == "recovered"
    assert diagnostic["effective_n"] == 8
    assert diagnostic["effective_second_order"] is True
    assert all(np.all(np.isfinite(metric.values)) for metric in result.metrics)


@pytest.mark.parametrize("missingKey", ["N", "secondOrder"])
def testSobolWarnsForMissingFieldsAndRecoversCompleteBlocks(missingKey):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x, meta = SaltelliDesign().sampleWithMeta(problem, 8, seed=17)
    del meta[missingKey]
    with pytest.warns(RuntimeWarning, match=f"Sobol.*{missingKey}"):
        result = Sobol(verboseFlag=False).analyze(problem, x, x[:, :1], meta)
    assert result.extra["sobol_design"]["status"] == "recovered"
    assert result.extra["sobol_design"]["effective_n"] == 8
    assert result.extra["sobol_design"]["effective_second_order"] is False
    assert missingKey not in result.meta


def testSobolWarnsForWrongMetadataEvenForConstantOutputs():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 128, seed=17)
    with pytest.warns(RuntimeWarning, match="Sobol.*blockSize"):
        result = Sobol(verboseFlag=False).analyze(problem, x, np.zeros((len(x), 1)), dict(meta, secondOrder=False))
    assert result.extra["sobol_design"]["status"] == "recovered"
    assert result.extra["sobol_design"]["effective_second_order"] is True
    for metric in result.metrics:
        np.testing.assert_array_equal(metric.values, 0)


@pytest.mark.parametrize("secondOrder", [False, True])
def testSobolValidNumpyMetadataWithoutOptionalBlockSizeKeepsAnalyticalEffects(secondOrder):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1] + 2 * x[:, 1:2])
    x, originalMeta = SaltelliDesign(secondOrder=secondOrder).sampleWithMeta(problem, 1024, seed=17)
    meta = {key: value for key, value in originalMeta.items() if key != "blockSize"}
    meta.update(N=np.int64(1024), secondOrder=np.bool_(secondOrder))
    result = Sobol(verboseFlag=False).analyze(problem, x, meta=meta)
    np.testing.assert_allclose(result["S1"].values, [[0.2, 0.8]], atol=0.003, rtol=0)
    np.testing.assert_allclose(result["ST"].values, [[0.2, 0.8]], atol=0.003, rtol=0)
    if secondOrder:
        np.testing.assert_allclose(result["S2"].values, [[0.0]], atol=0.003, rtol=0)
    assert "blockSize" not in meta
    assert "blockSize" not in result.meta
    assert meta["N"] == 1024 and meta["secondOrder"] == secondOrder
    assert result.extra["sobol_design"]["status"] == "validated"


def testSobolAmbiguousConflictingLayoutsWarnAndRemainUnavailable():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = np.zeros((12, 2))
    y = np.arange(12, dtype=float)[:, None]
    # Both block lengths match these constant inputs; the metadata conflicts.
    meta = dict(designType="saltelli", N=2, secondOrder=True, blockSize=4)
    with pytest.warns(RuntimeWarning, match="zero placeholders"):
        result = Sobol(verboseFlag=False).analyze(problem, x, y, meta)
    diagnostic = result.extra["sobol_design"]
    assert diagnostic["status"] == "not_estimated"
    assert diagnostic["effective_n"] is None
    assert diagnostic["effective_second_order"] is None
    for metric in result.metrics:
        np.testing.assert_array_equal(metric.values, 0)
    np.testing.assert_array_equal(result.Y, y)


@pytest.mark.parametrize("provideY", [False, True])
def testSobolUnavailableSelectedOutputsAndDiagnosticsPersistWithoutModelCalls(tmp_path, provideY):
    calls = []

    def objective(samples):
        calls.append(len(samples))
        return np.repeat(samples[:, :1], 3, axis=1)

    problem = Problem(nInput=2, nObj=3, lb=0, ub=1, objFunc=objective, objLabels=["first", "second", "third"])
    problem.workDir = str(tmp_path)
    x, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 8, seed=17)
    x = x[:-1]
    y = np.repeat(x[:, :1], 3, axis=1) if provideY else None
    with pytest.warns(RuntimeWarning, match="zero placeholders"):
        result = Sobol(verboseFlag=False, saveFlag=True).analyze(problem, x, y, meta, index=[2, 0])
    assert calls == []
    assert result["S1"].rowLabels == ["third", "first"]
    assert result["S1"].values.shape == (2, 2)
    with AnaReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
        assert reader.get_run_summary()["status"] == "finished"
    assert loaded.extra["sobol_design"] == result.extra["sobol_design"]
    assert loaded.extra["sobol_design"]["status"] == "not_estimated"
    if provideY:
        np.testing.assert_array_equal(loaded.Y, y[:, [2, 0]])
    else:
        assert loaded.Y is None


def testSobolRecoveredDesignAndOriginalMetadataPersist(tmp_path):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1] + 2 * x[:, 1:2])
    problem.workDir = str(tmp_path)
    x, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 128, seed=17)
    wrongMeta = dict(meta, secondOrder=False)
    with pytest.warns(RuntimeWarning, match="Sobol"):
        result = Sobol(verboseFlag=False, saveFlag=True).analyze(problem, x, meta=wrongMeta)
    with AnaReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
    assert loaded.meta["secondOrder"] is False
    assert loaded.settings["secondOrder"] is True
    assert loaded.extra["sobol_design"] == result.extra["sobol_design"]
    assert loaded.extra["sobol_design"]["effective_second_order"] is True
    np.testing.assert_array_equal(loaded["S1"].values, result["S1"].values)


def testSobolDivisibleRowsWithoutSaltelliStructureAreNotEstimated():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = np.random.default_rng(17).random((48, 2))
    meta = dict(designType="saltelli", N=1, secondOrder=True, blockSize=6)
    with pytest.warns(RuntimeWarning, match="zero placeholders"):
        result = Sobol(verboseFlag=False).analyze(problem, x, x[:, :1], meta)
    assert result.extra["sobol_design"]["status"] == "not_estimated"
    assert result.extra["sobol_design"]["metrics_available"] is False
