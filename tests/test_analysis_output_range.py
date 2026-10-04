"""Normalized screening survives physical squared-output underflow."""

import warnings

import numpy as np
import pytest

from UQPyL.analysis import DeltaTest, MARS
from UQPyL.doe import LHS
from UQPyL.problem import Problem


QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


@pytest.mark.parametrize("methodClass", [DeltaTest, MARS])
@pytest.mark.parametrize("factor", [1e-6, 1e-160, 1e-200, -1e-200, 1e150])
def testNormalizedImportanceSurvivesExtremeOutputUnits(methodClass, factor):
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = LHS("classic").sample(p, 128, seed=17)
    y = (3 * x[:, 0] + 1)[:, None]
    savedY = y.copy() * factor
    method = methodClass(**QUIET)
    baseline = method.analyze(p, x, y)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", RuntimeWarning)
        scaled = method.analyze(p, x, savedY)
    np.testing.assert_allclose(scaled["S1_norm"].values, baseline["S1_norm"].values, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(scaled.Y, savedY)
    assert np.all(np.isfinite(scaled["S1"].values))
    assert not any("no positive" in str(item.message) for item in emitted)
    if abs(factor) == 1e-200:
        np.testing.assert_array_equal(scaled["S1"].values, [[0, 0]])
        assert any("underflow" in str(item.message) for item in emitted)
    else:
        assert scaled["S1"].values[0, 0] > 0


@pytest.mark.parametrize("methodClass", [DeltaTest, MARS])
@pytest.mark.parametrize("factor", [1e155, 1e200])
def testUnrepresentableRawImportanceRaisesExplicitRangeError(methodClass, factor):
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = LHS("classic").sample(p, 128, seed=17)
    y = (3 * x[:, 0] + 1)[:, None] * factor
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="finite.*range"):
            methodClass(**QUIET).analyze(p, x, y)


@pytest.mark.parametrize("methodClass", [DeltaTest, MARS])
def testEachOutputKeepsItsOwnScaleAndConstantDefinition(methodClass):
    p = Problem(nInput=2, nObj=3, lb=0, ub=1, objFunc=lambda x: np.column_stack((x[:, 0], x[:, 1], x[:, 0])))
    x = LHS("classic").sample(p, 128, seed=17)
    y = np.column_stack((3 * x[:, 0] + 1, (2 * x[:, 1] + 1) * 1e-200, np.full(len(x), 1e-200)))
    with pytest.warns(RuntimeWarning, match="underflow"):
        result = methodClass(**QUIET).analyze(p, x, y)
    assert np.argmax(result["S1_norm"].values[0]) == 0
    assert np.argmax(result["S1_norm"].values[1]) == 1
    np.testing.assert_array_equal(result["S1_norm"].values[2], [0, 0])
    np.testing.assert_array_equal(result["S1"].values[1:], np.zeros((2, 2)))


@pytest.mark.parametrize("methodClass", [DeltaTest, MARS])
def testHugeFiniteConstantOutputsStayZeroWithoutWarnings(methodClass):
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = LHS("classic").sample(p, 128, seed=17)
    with np.errstate(over="raise", invalid="raise"):
        result = methodClass(**QUIET).analyze(p, x, np.full((len(x), 1), 1e308))
    np.testing.assert_array_equal(result["S1"].values, [[0, 0]])
    np.testing.assert_array_equal(result["S1_norm"].values, [[0, 0]])


@pytest.mark.parametrize("methodClass", [DeltaTest, MARS])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def testNonfiniteConstantOutputsAreRejectedBeforeConstantDetection(methodClass, value):
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = LHS("classic").sample(p, 128, seed=17)
    with pytest.raises(ValueError, match="finite output"):
        methodClass(**QUIET).analyze(p, x, np.full((len(x), 1), value))


def testMarsOutputScalingFitsOnlyTrainingRows():
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = LHS("classic").sample(p, 128, seed=17)
    y = (3 * x[:, 0] + 1)[:, None]
    baseline = MARS(**QUIET).analyze(p, x, y)
    validationRows = np.random.default_rng(0).permutation(len(x))[: len(x) // 5]
    changed = y.copy()
    changed[validationRows] = 10 * changed[validationRows] + 100
    with pytest.warns(RuntimeWarning, match="validation R2"):
        result = MARS(**QUIET).analyze(p, x, changed)
    np.testing.assert_array_equal(result["S1"].values, baseline["S1"].values)
    np.testing.assert_array_equal(result["S1_norm"].values, baseline["S1_norm"].values)
    before = baseline.extra["mars_validation"]["outputs"][0]
    after = result.extra["mars_validation"]["outputs"][0]
    assert before["scale_mantissa"] == after["scale_mantissa"]
    assert before["scale_exponent"] == after["scale_exponent"]


@pytest.mark.parametrize("value, exponent", [(1e-200, 700), (1e200, -700)])
def testSquaredUnitRestorationAvoidsIntermediateRangeLoss(value, exponent):
    from decimal import Decimal

    from UQPyL.analysis.methods._variance import restoreSquaredOutput

    # Both final values are representable, while explicitly forming the
    # squared physical scale would overflow/underflow before multiplication.
    expected = float(Decimal(str(value)) * Decimal(2) ** (2 * exponent))
    with np.errstate(over="raise", under="raise", invalid="raise"):
        actual, underflow = restoreSquaredOutput([value, -value], (1.0, exponent), "DeltaTest")
    np.testing.assert_allclose(actual, [expected, -expected], rtol=2e-15, atol=0)
    assert not underflow
