import importlib
import warnings

import numpy as np
import pytest

from UQPyL.doe import LHS
from UQPyL.problem import Problem

lhsModule = importlib.import_module("UQPyL.doe.methods.lhs")


def makeProblem(nInput):
    return Problem(nInput=nInput, nObj=1, lb=-3., ub=7.,
                   objFunc=lambda X: np.sum(X, axis=1, keepdims=True))


def correlationScore(samples):
    centered = samples - samples.mean(axis=0)
    normalized = centered / np.sqrt(np.sum(centered ** 2, axis=0))
    # Independent pairwise dot products, without np.corrcoef.
    return max(abs(normalized[:, i] @ normalized[:, j])
               for i in range(samples.shape[1]) for j in range(i))


def assertLatin(samples):
    nSamples, nInput = samples.shape
    assert np.all(np.isfinite(samples))
    assert np.all((samples >= 0) & (samples < 1))
    bins = np.floor(samples * nSamples).astype(int)
    for index in range(nInput):
        np.testing.assert_array_equal(np.sort(bins[:, index]), np.arange(nSamples))


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("nSamples,nInput", [(12, 2), (60, 3), (12, 6)])
def testSelectsLeastCorrelatedCandidateColumns(seed, nSamples, nInput, capsys):
    rng = np.random.default_rng(seed)
    candidates = [lhsModule._lhs_classic(nSamples, nInput, rng) for _ in range(8)]
    scores = [correlationScore(candidate) for candidate in candidates]
    samples = LHS("correlation", iterations=8).sample(makeProblem(nInput), nSamples, seed=seed, output="unit")
    np.testing.assert_array_equal(samples, candidates[np.argmin(scores)])
    assert correlationScore(samples) <= scores[0] + 1e-14
    assertLatin(samples)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("nSamples,nInput", [(1, 1), (10, 1), (1, 4)])
def testDegenerateCasesUseClassicLhsWithoutCorrelation(nSamples, nInput, monkeypatch):
    problem = makeProblem(nInput)
    expected = LHS("classic").sample(problem, nSamples, seed=1, output="unit")

    def unexpectedCorrelation(*args, **kwargs):
        raise AssertionError("No correlation is defined for this case")

    monkeypatch.setattr(lhsModule.np, "corrcoef", unexpectedCorrelation)
    samples = LHS("correlation").sample(problem, nSamples, seed=1, output="unit")
    np.testing.assert_array_equal(samples, expected)
    assertLatin(samples)


@pytest.mark.parametrize("nInput", [2, 5])
def testTwoSampleCorrelationStillReturnsValidLatinDesign(nInput, capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        samples = LHS("correlation").sample(makeProblem(nInput), 2, seed=1, output="unit")
    assertLatin(samples)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("sign", [1, -1])
def testPerfectOffDiagonalCorrelationIsIncludedInScore(sign, monkeypatch):
    first = np.array([.1, .3, .5, .7, .9])
    correlated = np.column_stack((first, first if sign == 1 else first[::-1]))
    better = np.column_stack((first, first[[1, 4, 0, 3, 2]]))
    candidates = iter([correlated, better])
    monkeypatch.setattr(lhsModule, "_lhs_classic", lambda *args: next(candidates))
    samples = LHS("correlation", iterations=2).sample(makeProblem(2), 5, seed=1, output="unit")
    np.testing.assert_array_equal(samples, better)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True, None, "3"])
def testInvalidCorrelationIterationsFailClearly(iterations):
    with pytest.raises(ValueError, match="iterations.*positive integer"):
        LHS("correlation", iterations=iterations).sample(makeProblem(2), 10, seed=1)


def testRealAndUnitOutputsReproduceWithoutChangingGlobalRng():
    problem = makeProblem(3)
    method = LHS("correlation", iterations=np.int64(4))
    globalBefore = np.random.get_state()
    unit, meta = method.sampleWithMeta(problem, 20, seed=7, output="unit")
    real = method.sample(problem, 20, seed=7)
    repeated = method.sample(problem, 20, seed=7, output="unit")
    np.testing.assert_array_equal(unit, repeated)
    np.testing.assert_array_equal(real, problem.unit_to_space(unit))
    globalAfter = np.random.get_state()
    for before, after in zip(globalBefore, globalAfter):
        np.testing.assert_equal(before, after)
    assert meta["criterion"] == "correlation" and meta["iterations"] == 4
    assert meta["output"] == "unit"
    assertLatin(unit)


def testCorrelationMatrixUsesVariableDimensions(monkeypatch):
    original = np.corrcoef
    shapes = []

    def recordCorrelation(*args, **kwargs):
        result = original(*args, **kwargs)
        shapes.append(result.shape)
        return result

    monkeypatch.setattr(lhsModule.np, "corrcoef", recordCorrelation)
    LHS("correlation", iterations=3).sample(makeProblem(3), 100, seed=1)
    assert shapes == [(3, 3)] * 3
