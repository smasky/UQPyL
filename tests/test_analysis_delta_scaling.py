"""Independent distance references and signed DeltaTest score contracts."""

from itertools import product

import numpy as np
import pytest

from analysis_test_support import pairwiseDelta

from UQPyL.analysis import DeltaTest
from UQPyL.doe import LHS
from UQPyL.problem import Problem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def makeData():
    p = Problem(nInput=3, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])
    x = LHS("classic").sample(p, 128, seed=17)
    y = (x[:, 0] + 2 * x[:, 1])[:, None]
    unit = (x + 1) / 2
    base = pairwiseDelta(unit, y)
    scores = np.array([pairwiseDelta(np.delete(unit, j, axis=1), y) - base for j in range(3)])
    return x, y, unit, scores


@pytest.mark.parametrize("factor", [1, 0.001, 1000])
@pytest.mark.numerical
def testDeltaUnitsMatchIndependentScaledDistanceReference(factor):
    x, y, _, expected = makeData()
    scale, shift = np.array([factor, 1, 1]), np.array([10, -3, 20])
    p = Problem(nInput=3, nObj=1, lb=-scale + shift, ub=scale + shift, objFunc=lambda x: x[:, :1])
    real = x * scale + shift
    savedX, savedY = real.copy(), y.copy()
    result = DeltaTest(**QUIET).analyze(p, real, y)
    np.testing.assert_allclose(result["S1"].values[0], expected, atol=1e-14)
    np.testing.assert_allclose(result["S1_norm"].values[0], expected / np.sum(abs(expected)), atol=1e-14)
    np.testing.assert_array_equal(real, savedX)
    np.testing.assert_array_equal(y, savedY)
    np.testing.assert_array_equal(result.X, real)


@pytest.mark.parametrize("scores", [[-3, 1, -1], [2, -2, 0], [-1, -2, -3]])
def testDeltaSignedNormalizationPreservesOrderAndCancellation(monkeypatch, scores):
    p = Problem(nInput=3, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = np.random.default_rng(7).random((10, 3))
    # Isolate reductions with controlled Delta values; the independent pairwise
    # tests above exercise the actual estimator and distance preprocessing.
    values = iter([10] + [10 + s for s in scores])
    method = DeltaTest(**QUIET)
    monkeypatch.setattr(method, "_cal_delta", lambda *args: next(values))
    # A unit output span isolates signed reduction/normalization from the
    # physical-scale restoration, whose numerical behavior is tested separately.
    y = np.linspace(0, 1, len(x))[:, None]
    if max(scores) <= 0:
        with pytest.warns(RuntimeWarning, match="no positive"):
            result = method.analyze(p, x, y)
    else:
        result = method.analyze(p, x, y)
    np.testing.assert_array_equal(result["S1"].values, [scores])
    np.testing.assert_allclose(result["S1_norm"].values[0], np.array(scores) / sum(abs(np.array(scores))))
    assert np.argmax(result["S1_norm"].values[0]) == np.argmax(scores)


def testDeltaFixedAndSampleConstantColumnsHaveZeroContribution():
    rng = np.random.default_rng(22)
    active = rng.random((40, 2))
    x = np.column_stack((np.full(40, 5.0), active, np.full(40, 7.0)))
    y = (active[:, 0] + 2 * active[:, 1])[:, None]
    p = Problem(nInput=4, nObj=1, lb=[5, 0, 0, 0], ub=[5, 1, 1, 10], objFunc=lambda x: x[:, :1])
    base = pairwiseDelta(active, y)
    expected = [0, pairwiseDelta(active[:, 1:2], y) - base, pairwiseDelta(active[:, :1], y) - base, 0]
    with np.errstate(divide="raise", invalid="raise"):
        result = DeltaTest(**QUIET).analyze(p, x, y)
    np.testing.assert_allclose(result["S1"].values[0], expected, atol=1e-14)


def testDeltaNonconstantOutputsWithoutPositiveScoresWarnAndReturn():
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = np.zeros((3, 2))
    y = np.array([[0.0], [1.0], [4.0]])
    with pytest.warns(RuntimeWarning, match="no positive"):
        result = DeltaTest(**QUIET).analyze(p, x, y)
    np.testing.assert_array_equal(result["S1_norm"].values, [[0, 0]])
    # Constant outputs are defined zeros, with no warning under -W error.
    result = DeltaTest(**QUIET).analyze(p, x, np.full_like(y, 4))
    np.testing.assert_array_equal(result["S1_norm"].values, [[0, 0]])


@pytest.mark.parametrize("factor", [0.001, 1000])
@pytest.mark.parametrize("entry", ["vio", "ea"])
def testDeltaSubsetSearchUsesSameScaledDistances(monkeypatch, factor, entry):
    x, y, unit, _ = makeData()
    scale = np.array([factor, 1, 1])
    p = Problem(nInput=3, nObj=1, lb=-scale, ub=scale, objFunc=lambda x: x[:, :1])
    masks = np.array(list(product([0, 1], repeat=3)))
    expected = np.array([pairwiseDelta(unit[:, mask.astype(bool)], y) if np.any(mask) else np.inf for mask in masks])
    if entry == "vio":
        labels = DeltaTest(**QUIET).findCombVio(p, x * scale, y)
        assert labels == [p.xLabels[j] for j in np.flatnonzero(masks[np.argmin(expected)])]
    else:
        import UQPyL.optimization.soea as optimization

        class ExhaustiveGA:
            def __init__(self, **kwargs):
                pass

            def run(self, selectionProblem):
                values = selectionProblem.evaluate(masks).objs[:, 0]
                assert np.isinf(values[0])
                np.testing.assert_allclose(values[1:] / values[1:].sum(), expected[1:] / expected[1:].sum(), atol=1e-14)
                return int(np.argmin(values))

        monkeypatch.setattr(optimization, "GA", ExhaustiveGA)
        best = DeltaTest(**QUIET).findCombEA(p, x * scale, y, FEs=8, verboseFlag=False, saveFlag=False)
        assert best == int(np.argmin(expected))


@pytest.mark.parametrize("providedY", [False, True])
def testDeltaUnitMetadataAndEvaluationUseRealCoordinates(providedY):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return x[:, :1] + 2 * x[:, 1:2]

    p = Problem(nInput=2, nObj=1, lb=[10, -3], ub=[100, 2], objFunc=objective)
    real, realMeta = LHS().sampleWithMeta(p, 128, seed=17)
    unit, unitMeta = LHS().sampleWithMeta(p, 128, seed=17, output="unit")
    y = objective(real) if providedY else None
    expected = DeltaTest(**QUIET).analyze(p, real, y, realMeta)
    calls.clear()
    actual = DeltaTest(**QUIET).analyze(p, unit, y, unitMeta)
    np.testing.assert_allclose(actual["S1"].values, expected["S1"].values, atol=1e-12)
    np.testing.assert_allclose(actual.X, real)
    assert unitMeta["output"] == "unit"
    if providedY:
        assert calls == []
    else:
        assert len(calls) == 1
        np.testing.assert_allclose(calls[0], real)


@pytest.mark.parametrize("factor", [0.001, 1000])
@pytest.mark.numerical
def testDeltaNumericDiscreteChoicesUsePhysicalValueRange(factor):
    rng = np.random.default_rng(14)
    choices = np.array([10.0, 15.0, 30.0])
    selected = rng.choice(choices, 40)
    continuous = rng.uniform(-1, 1, 40)
    x = np.column_stack((selected * factor + 7, continuous))
    # Removing the continuous axis creates exact nearest-neighbor ties.
    # Equal outputs within each choice make their contribution independent of
    # which tied rows KDTree and the pairwise reference select.
    y = (selected / 20)[:, None]
    p = Problem(
        nInput=2,
        nObj=1,
        lb=[0, -1],
        ub=[1, 1],
        varType=[2, 0],
        varSet={0: (choices * factor + 7).tolist()},
        objFunc=lambda x: x[:, :1],
    )
    distanceX = np.column_stack(((selected - 10) / 20, (continuous + 1) / 2))
    base = pairwiseDelta(distanceX, y)
    expected = [pairwiseDelta(distanceX[:, 1:2], y) - base, pairwiseDelta(distanceX[:, :1], y) - base]
    result = DeltaTest(**QUIET).analyze(p, x, y)
    np.testing.assert_allclose(result["S1"].values[0], expected, atol=1e-14)


def testDeltaEmptySamplesStillRaiseSampleCountError():
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    with pytest.raises(ValueError, match="smaller than the sample count"):
        DeltaTest(**QUIET).analyze(p, np.empty((0, 2)), np.empty((0, 1)))
