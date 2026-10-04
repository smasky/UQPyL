"""Independent numerical references, not just shape or self-consistency checks."""

import numpy as np
import pytest

from analysis_test_support import pairwiseDelta

from UQPyL.analysis import DeltaTest, Morris, RSA, Sobol, FAST, RBDFAST, MARS
from UQPyL.doe import SaltelliDesign, FASTDesign, LHS, MorrisDesign
from UQPyL.problem import Problem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


@pytest.mark.parametrize("neighbors", [1, 2, 3])
@pytest.mark.numerical
def testDeltaMagnitudeMatchesIndependentPairwiseDefinition(neighbors):
    X = np.random.default_rng(91).random((12, 3))
    Y = (X[:, 0] + 2 * X[:, 1] ** 2)[:, None]
    method = DeltaTest(nNeighbors=neighbors, **QUIET)
    assert method._cal_delta(X, Y, neighbors) == pytest.approx(pairwiseDelta(X, Y, neighbors), rel=1e-13)
    problem = Problem(nInput=3, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    expected = [pairwiseDelta(np.delete(X, j, axis=1), Y, neighbors) - pairwiseDelta(X, Y, neighbors) for j in range(3)]
    np.testing.assert_allclose(method.analyze(problem, X, Y)["S1"].values[0], expected, rtol=1e-12, atol=1e-15)


@pytest.mark.numerical
def testDeltaDuplicateInputsStillExcludeTheQueryRow():
    X = np.zeros((3, 2))
    Y = np.array([[0.0], [1.0], [4.0]])
    assert DeltaTest(**QUIET)._cal_delta(X, Y, 2) == pytest.approx(13 / 3)


@pytest.mark.numerical
def testDeltaNormalizedImportanceDoesNotDependOnOutputUnits():
    p = Problem(nInput=3, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1] + 2 * x[:, 1:2])
    X = LHS("classic").sample(p, 128, seed=31)
    Y = p.evaluate(X).objs
    method = DeltaTest(**QUIET)
    original = method.analyze(p, X, Y)
    scaled = method.analyze(p, X, Y * 1e-6)
    np.testing.assert_allclose(scaled["S1"].values, original["S1"].values * 1e-12, rtol=1e-12, atol=0)
    np.testing.assert_allclose(scaled["S1_norm"].values, original["S1_norm"].values, rtol=1e-12)


@pytest.mark.parametrize("methodClass", [Sobol, FAST, RBDFAST])
@pytest.mark.parametrize("kind", ["linear", "product"])
@pytest.mark.numerical
def testVarianceIndicesMatchAnalyticMainAndInteractionEffects(methodClass, kind):
    objective = (
        (lambda x: (x[:, 0] + 2 * x[:, 1])[:, None]) if kind == "linear" else (lambda x: (x[:, 0] * x[:, 1])[:, None])
    )
    p = Problem(nInput=3, nObj=1, lb=-1, ub=1, objFunc=objective)
    if methodClass is Sobol:
        X, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(p, 4096, seed=41)
    elif methodClass is FAST:
        X, meta = FASTDesign(M=4).sampleWithMeta(p, 2049, seed=41)
    else:
        X, meta = LHS("classic").sample(p, 8192, seed=41), None
    result = methodClass(**QUIET).analyze(p, X, meta=meta)
    np.testing.assert_allclose(
        result["S1"].values[0], [0.2, 0.8, 0] if kind == "linear" else [0, 0, 0], atol=0.035, rtol=0
    )
    if methodClass is not RBDFAST:
        np.testing.assert_allclose(
            result["ST"].values[0], [0.2, 0.8, 0] if kind == "linear" else [1, 1, 0], atol=0.035, rtol=0
        )
    if methodClass is Sobol:
        np.testing.assert_allclose(
            result["S2"].values[0], [0, 0, 0] if kind == "linear" else [1, 0, 0], atol=0.035, rtol=0
        )


@pytest.mark.parametrize("numLevels", [4, 6])
@pytest.mark.numerical
def testMorrisReportsStandardEffectsOnUnequalBounds(numLevels):
    p = Problem(
        nInput=3, nObj=1, lb=[10, -2, 100], ub=[30, 1, 105], objFunc=lambda x: (3 * x[:, 0] - 2 * x[:, 1])[:, None]
    )
    X, meta = MorrisDesign(numLevels=numLevels).sampleWithMeta(p, 20, seed=19)
    result = Morris(**QUIET).analyze(p, X, meta=meta)
    np.testing.assert_allclose(result["mu"].values, [[60, -6, 0]], atol=1e-12)
    np.testing.assert_allclose(result["mu_star"].values, [[60, 6, 0]], atol=1e-12)
    np.testing.assert_allclose(result["sigma"].values, 0, atol=1e-12)
    np.testing.assert_allclose(result["S1_norm"].values, [[10 / 11, 1 / 11, 0]], atol=1e-12)


@pytest.mark.numerical
def testRsaMatchesHandDerivedRankStatistic():
    # Two disjoint groups of four: pooled ranks 1..4 versus 5..8.
    # U = sum((r_i-i)^2) * 4 + sum((s_j-j)^2) * 4 = 256.
    # T = U/(nm(n+m)) - (4nm-1)/(6(n+m)) = 11/16.
    X = np.column_stack((np.arange(8.0), [0, 1, 2, 3, 0, 1, 2, 3]))
    Y = np.array([0] * 4 + [1] * 4)[:, None]
    p = Problem(nInput=2, nObj=1, lb=0, ub=8, objFunc=lambda x: x[:, :1])
    result = RSA(nRegion=2, **QUIET).analyze(p, X, Y)
    # Equal input distributions in the second dimension have T=0, including ties.
    np.testing.assert_allclose(result["S1"].values, [[11 / 16, 0]], atol=1e-14)


@pytest.mark.numerical
def testMarsIdentifiesSingleActiveInputWithoutCallingItVarianceShare():
    p = Problem(nInput=3, nObj=1, lb=0, ub=1, objFunc=lambda x: (3 * x[:, 0] + 1)[:, None])
    X = LHS("classic").sample(p, 512, seed=17)
    result = MARS(**QUIET).analyze(p, X)
    scores = result["S1"].values[0]
    assert scores[0] > 0
    assert scores[0] > 100 * max(scores[1:])
    assert result["S1_norm"].values[0, 0] > 0.99


@pytest.mark.numerical
def testMarsNormalizedImportanceDoesNotDependOnOutputUnits():
    p = Problem(nInput=3, nObj=1, lb=0, ub=1, objFunc=lambda x: (3 * x[:, 0] + 1)[:, None])
    X = LHS("classic").sample(p, 512, seed=17)
    Y = p.evaluate(X).objs
    method = MARS(**QUIET)
    original = method.analyze(p, X, Y)
    scaled = method.analyze(p, X, Y * 1e-6)
    # Adaptive hinge selection can take a different near-tied path after
    # floating-point rescaling, particularly in the reduced noise-only fit.
    # Check squared units to 1%; the meaningful normalized ranking stays exact.
    np.testing.assert_allclose(scaled["S1"].values, original["S1"].values * 1e-12, rtol=0.01, atol=1e-25)
    np.testing.assert_allclose(scaled["S1_norm"].values, original["S1_norm"].values, atol=1e-10)
