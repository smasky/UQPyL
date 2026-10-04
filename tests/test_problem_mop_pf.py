import numpy as np
import pytest

from UQPyL.problem.mop import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6


def nondominatedMask(points):
    # Independent pairwise definition, without the production NDSort.
    result = np.ones(len(points), dtype=bool)
    for start in range(0, len(points), 64):
        block = points[start : start + 64]
        dominates = np.all(points[:, None, :] <= block[None, :, :], axis=2)
        dominates &= np.any(points[:, None, :] < block[None, :, :], axis=2)
        result[start : start + len(block)] = ~np.any(dominates, axis=0)
    return result


@pytest.mark.parametrize("problemClass", [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6])
def testZdtFrontSatisfiesAnalyticRelation(problemClass):
    pf = problemClass().getPF()
    assert isinstance(pf, tuple) and len(pf) == 2
    x, y = np.broadcast_arrays(*pf)
    assert x.size > 0 and not np.any(np.isinf(x)) and not np.any(np.isinf(y))
    finite = np.isfinite(x) & np.isfinite(y)
    assert np.any(finite)
    if problemClass is ZDT3:
        # NaNs intentionally break the curve between disconnected pieces.
        np.testing.assert_array_equal(np.isnan(x), np.isnan(y))
        grid = np.linspace(0, 1, 300)
        fullY = 1 - np.sqrt(grid) - grid * np.sin(10 * np.pi * grid)
        np.testing.assert_array_equal(finite, nondominatedMask(np.column_stack((grid, fullY))))
        expected = 1 - np.sqrt(x[finite]) - x[finite] * np.sin(10 * np.pi * x[finite])
    else:
        assert np.all(finite)
        expected = 1 - x**2 if problemClass in (ZDT2, ZDT6) else 1 - np.sqrt(x)
    np.testing.assert_allclose(y[finite], np.asarray(expected).ravel(), atol=1e-12)
    assert np.all((x[finite] >= 0) & (x[finite] <= 1))
    assert x[finite].min() < 0.3 and x[finite].max() > 0.8


@pytest.mark.parametrize("problemClass", [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7])
def testDtlzFrontSatisfiesGeometry(problemClass):
    pf = problemClass(nInput=7, nObj=3).getPF()
    assert isinstance(pf, tuple) and len(pf) == 3
    arrays = np.broadcast_arrays(*pf)
    points = np.column_stack([array.ravel() for array in arrays])
    assert len(points) > 0 and not np.any(np.isinf(points))
    if problemClass is DTLZ7:
        x, y, z = points.T
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        expectedZ = 6 - x * (1 + np.sin(3 * np.pi * x)) - y * (1 + np.sin(3 * np.pi * y))
        finite = np.isfinite(z)
        assert np.any(finite) and np.any(~finite)
        np.testing.assert_allclose(z[finite], expectedZ[finite], atol=1e-12)
        np.testing.assert_array_equal(finite, nondominatedMask(np.column_stack((x, y, expectedZ))))
        assert np.all(points[finite] >= 0)
        return
    assert np.all(np.isfinite(points)) and np.all(points >= 0)
    if problemClass is DTLZ1:
        np.testing.assert_allclose(points.sum(axis=1), 0.5, atol=1e-12)
        np.testing.assert_allclose(points.max(axis=0), 0.5, atol=1e-12)
    else:
        np.testing.assert_allclose(np.sum(points**2, axis=1), 1, atol=1e-12)
        if problemClass in (DTLZ5, DTLZ6):
            np.testing.assert_allclose(points[:, 0], points[:, 1], atol=1e-12)
            np.testing.assert_allclose(points.max(axis=0), [1 / np.sqrt(2), 1 / np.sqrt(2), 1], atol=1e-12)
        else:
            np.testing.assert_allclose(points.max(axis=0), 1, atol=1e-12)


@pytest.mark.parametrize(
    "problemClass,nObj",
    [
        *[(cls, 3) for cls in [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]],
        *[(cls, 2) for cls in [DTLZ2, DTLZ4, DTLZ5, DTLZ6, DTLZ7]],
    ],
)
def testFrontProblemsRejectUnsupportedObjectiveCounts(problemClass, nObj):
    with pytest.raises(ValueError):
        problemClass(nInput=10, nObj=nObj)
