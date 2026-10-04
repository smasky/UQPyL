"""Subset ranking retains physical output weighting under extreme common units."""

from itertools import product
import warnings

import numpy as np
import pytest

from UQPyL.analysis import DeltaTest
from UQPyL.doe import LHS
from UQPyL.problem import Problem


@pytest.mark.parametrize("entry", ["vio", "ea"])
@pytest.mark.parametrize("factor", [1, 1e-200, -1e-200, 1e200])
def testDeltaSubsetSelectionKeepsActiveVariableUnderCommonOutputScaling(entry, factor):
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    x = LHS("classic").sample(p, 128, seed=17)
    y = (3 * x[:, 0] + 1)[:, None] * factor
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", RuntimeWarning)
        method = DeltaTest(verboseFlag=False)
        if entry == "vio":
            assert method.findCombVio(p, x, y) == [p.xLabels[0]]
        else:
            result = method.findCombEA(p, x, y, FEs=50, verboseFlag=False, saveFlag=False, seed=17)
            np.testing.assert_array_equal(result.bestDecs, [[1, 0]])
            assert np.all(np.isfinite(result.bestObjs))
            assert result.extra["delta_selection"]["objective_units"] == "scaled_output_squared"
            assert all(np.isfinite(result.history.bestObjHistory))
    assert not any("encountered in square" in str(item.message) for item in emitted)


def testDeltaSubsetCommonScalePreservesMultiOutputRelativeWeight(monkeypatch):
    import UQPyL.optimization.soea as optimization

    p = Problem(nInput=2, nObj=2, lb=0, ub=1, objFunc=lambda x: x)
    x = LHS("classic").sample(p, 128, seed=17)
    y = np.column_stack((x[:, 0], 20 * x[:, 1]))
    masks = np.array(list(product([0, 1], repeat=2)))
    method = DeltaTest(verboseFlag=False)
    expected = np.array([method._cal_delta(x[:, mask.astype(bool)], y, 2) for mask in masks[1:]])

    class ExhaustiveGA:
        def __init__(self, **kwargs):
            pass

        def run(self, problem):
            actual = problem.evaluate(masks).objs[:, 0]
            assert np.isinf(actual[0])
            np.testing.assert_allclose(actual[1:] / actual[1:].sum(), expected / expected.sum(), atol=1e-14)
            return int(np.argmin(actual))

    monkeypatch.setattr(optimization, "GA", ExhaustiveGA)
    for factor in (1, 1e-200, 1e200):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            best = method.findCombEA(p, x, y * factor, FEs=4, verboseFlag=False, saveFlag=False)
        assert best == np.argmin(expected) + 1


@pytest.mark.parametrize("factor", [1.0, 1e-200, 1e200])
def testDeltaConstantLargeOutputColumnDoesNotEraseSmallActiveOutput(factor):
    p = Problem(nInput=2, nObj=2, lb=0, ub=1, objFunc=lambda x: x)
    x = LHS("classic").sample(p, 128, seed=17)
    y = np.column_stack(((3 * x[:, 0] + 1) * factor, np.full(len(x), 1e300)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert DeltaTest(verboseFlag=False).findCombVio(p, x, y) == [p.xLabels[0]]


def testDeltaScaledSelectionMetadataSurvivesSqliteRoundTrip(tmp_path):
    from UQPyL.optimization.runtime import OptReader

    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    p.workDir = str(tmp_path)
    x = LHS("classic").sample(p, 128, seed=17)
    result = DeltaTest(verboseFlag=False).findCombEA(p, x, x[:, :1], FEs=50, verboseFlag=False, saveFlag=True, seed=17)
    database = next(tmp_path.rglob("*.sqlite3"))
    with OptReader(str(database)) as reader:
        restored = reader.load_result()
    assert restored.extra["delta_selection"] == result.extra["delta_selection"]
    np.testing.assert_array_equal(restored.bestObjs, result.bestObjs)


def testSquaredRestorationHandlesTinyMantissaWithLargeExponent():
    from decimal import Decimal
    from UQPyL.analysis.methods._variance import restoreSquaredOutput

    expected = float((Decimal("1e-300") * Decimal(2) ** 1000) ** 2)
    actual, underflow = restoreSquaredOutput([1.0], (1e-300, 1000), "DeltaTest")
    assert actual[0] == pytest.approx(expected, rel=1e-14)
    assert not underflow
