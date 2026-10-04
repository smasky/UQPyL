"""Independent pairwise references for permutation-invariant tied neighbors."""

from itertools import product

import numpy as np
import pytest

from UQPyL.analysis import DeltaTest
from UQPyL.problem import Problem


QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def averagedDelta(x, y, k):
    distances = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    cutoffs = np.sort(distances, axis=1)[:, k - 1]
    rowErrors = []
    for row, cutoff in enumerate(cutoffs):
        tied = np.isclose(distances[row], cutoff, rtol=32 * np.finfo(float).eps * x.shape[1], atol=0)
        closer = (distances[row] < cutoff) & ~tied
        squared = np.mean((y[row] - y) ** 2, axis=1)
        rowErrors.append((squared[closer].sum() + (k - closer.sum()) * squared[tied].mean()) / k)
    return 0.5 * np.mean(rowErrors)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.numerical
def testDeltaDuplicateCutoffMatchesHandAverage(k):
    x = np.array([[0.0], [0.0], [0.0], [1.0]])
    y = np.array([[0.0], [1.0], [4.0], [7.0]])
    # Per-row averages: 17/2, 5, 25/2, 94/3. Self is excluded.
    expected = 43 / 6
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(x))
        assert DeltaTest(**QUIET)._cal_delta(x[order], y[order], k) == pytest.approx(expected)


@pytest.mark.numerical
def testDeltaCutoffWeightDoesNotReweightStrictlyCloserNeighbors():
    x = np.array([[0.0], [0.25], [1.0], [-1.0]])
    y = np.array([[0.0], [2.0], [5.0], [9.0]])
    # At x=0, one guaranteed neighbor has error 4; the second is chosen
    # uniformly between errors 25 and 81. The row mean is (4+53)/2.
    # Other row means are 6.5, 17, 65. Delta is half their overall mean.
    assert DeltaTest(**QUIET)._cal_delta(x, y, 2) == pytest.approx(117 / 8)


@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("duplicates", [False, True])
@pytest.mark.numerical
def testDeltaGridAndProjectionTiesMatchReferenceUnderPermutation(k, duplicates):
    x = np.array(list(product(np.arange(4) / 3, repeat=2)))
    if duplicates:
        x = np.repeat(x, 3, axis=0)
    y = (x[:, 0] + 1.2 * x[:, 1])[:, None]
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    base = averagedDelta(x, y, k)
    expected = np.array([averagedDelta(np.delete(x, j, axis=1), y, k) - base for j in range(2)])
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(x))
        result = DeltaTest(nNeighbors=k, **QUIET).analyze(p, x[order], y[order])
        np.testing.assert_allclose(result["S1"].values[0], expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(result["S1_norm"].values[0], expected / sum(abs(expected)), atol=1e-13)
        assert np.argmax(result["S1"].values[0]) == 1


@pytest.mark.parametrize("k", [1, 2, 6])
@pytest.mark.numerical
def testDeltaAllDuplicateRowsUseAllOtherOutputs(k):
    x = np.zeros((7, 2))
    y = np.column_stack((np.arange(7.0), np.arange(7.0) ** 2))
    expected = 7 / 6 * np.mean(np.var(y, axis=0))
    assert DeltaTest(**QUIET)._cal_delta(x, y, k) == pytest.approx(expected)


def testDeltaDistinctNearbyDistancesAreNotMerged():
    x = np.array([[0.0], [1.0], [-1.00000001], [3.0]])
    y = np.array([[0.0], [2.0], [100.0], [4.0]])
    assert DeltaTest(**QUIET)._cal_delta(x, y, 1) == pytest.approx(averagedDelta(x, y, 1))


def testDeltaLargeDuplicateGroupDoesNotQueryEveryBoundarySet(monkeypatch):
    import importlib

    module = importlib.import_module("UQPyL.analysis.methods.delta")
    realTree = module.KDTree

    class DuplicateTree:
        def __init__(self, x):
            self.tree = realTree(x)

        def query(self, *args, **kwargs):
            return self.tree.query(*args, **kwargs)

        def query_ball_point(self, *args, **kwargs):
            raise AssertionError("An identical-coordinate group should be aggregated once.")

    monkeypatch.setattr(module, "KDTree", DuplicateTree)
    x = np.zeros((257, 2))
    y = np.arange(len(x), dtype=float)[:, None]
    assert DeltaTest(**QUIET)._cal_delta(x, y, 2) == pytest.approx(len(x) / (len(x) - 1) * np.var(y))


@pytest.mark.parametrize("entry", ["ea", "vio"])
def testDeltaSubsetSearchUsesSamePermutationInvariantTieRule(monkeypatch, entry):
    x = np.array(list(product(np.arange(3) / 2, repeat=2)))
    y = (x[:, 0] + 1.2 * x[:, 1])[:, None]
    masks = np.array(list(product([0, 1], repeat=2)))
    expected = np.array([averagedDelta(x[:, mask.astype(bool)], y, 1) if np.any(mask) else np.inf for mask in masks])
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    if entry == "ea":
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
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(len(x))
        method = DeltaTest(nNeighbors=1, **QUIET)
        if entry == "ea":
            assert method.findCombEA(p, x[order], y[order], FEs=4, verboseFlag=False, saveFlag=False) == np.argmin(
                expected
            )
        else:
            labels = method.findCombVio(p, x[order], y[order])
            assert labels == [p.xLabels[j] for j in np.flatnonzero(masks[np.argmin(expected)])]
