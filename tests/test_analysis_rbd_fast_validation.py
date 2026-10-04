"""RBD-FAST must not turn unsupported samples into sensitivity evidence."""

import numpy as np
import pytest

from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS
from UQPyL.problem import Problem


@pytest.mark.parametrize("seed", [None, 0, 17, 41])
def testRbdFastFixedInputHasZeroEffectUnderRowPermutations(seed):
    problem = Problem(nInput=2, nObj=1, lb=[0, 0], ub=[0, 1], objFunc=lambda x: x[:, 1:2])
    x = np.column_stack([np.zeros(256), np.linspace(0, 1, 256)])
    if seed is not None:
        x = x[np.random.default_rng(seed).permutation(len(x))]
    result = RBDFAST(verboseFlag=False).analyze(problem, x, x[:, 1:2])
    np.testing.assert_allclose(result["S1"].values, [[0.0, 1.0]], atol=0.003, rtol=0)
    assert result["S1"].values[0, 0] == 0.0


@pytest.mark.parametrize("seed", [None, 17])
def testRbdFastRejectsNonconstantRepeatedInputValues(seed):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, varType=[1, 0], objFunc=lambda x: x[:, 1:2])
    x = np.column_stack([np.repeat([0, 1], 128), np.tile(np.linspace(0, 1, 128), 2)])
    if seed is not None:
        x = x[np.random.default_rng(seed).permutation(len(x))]
    with pytest.raises(ValueError, match="nonconstant.*repeated values"):
        RBDFAST(verboseFlag=False).analyze(problem, x, x[:, 1:2])


@pytest.mark.parametrize("count,harmonics", [(8, 4), (8, 5), (16, 12)])
def testRbdFastRejectsInvalidHarmonicSampleBudget(count, harmonics):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    x = np.random.default_rng(17).random((count, 2))
    with pytest.raises(ValueError, match=r"N > 2\*M"):
        RBDFAST(M=harmonics, verboseFlag=False).analyze(problem, x, x[:, 1:2])


@pytest.mark.parametrize("harmonics", [0, -1, True, 2.5])
def testRbdFastRequiresPositiveIntegerHarmonics(harmonics):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    x = np.random.default_rng(17).random((64, 2))
    with pytest.raises(ValueError, match="positive integer"):
        RBDFAST(M=harmonics, verboseFlag=False).analyze(problem, x, x[:, 1:2])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def testRbdFastRejectsNonfiniteInputs(invalid):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    x = np.random.default_rng(17).random((64, 2))
    x[0, 0] = invalid
    with pytest.raises(ValueError, match="finite input"):
        RBDFAST(verboseFlag=False).analyze(problem, x, x[:, 1:2])


def testRbdFastContinuousSamplePermutationPreservesAnalyticalRanking():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    x = LHS("classic").sample(problem, 1024, seed=17)
    y = x[:, 1:2]
    originalX, originalY = x.copy(), y.copy()
    expected = RBDFAST(M=np.int64(4), verboseFlag=False).analyze(problem, x, y)
    order = np.random.default_rng(3).permutation(len(x))
    actual = RBDFAST(verboseFlag=False).analyze(problem, x[order], y[order])
    np.testing.assert_allclose(expected["S1"].values, [[0, 1]], atol=0.003, rtol=0)
    np.testing.assert_allclose(actual["S1"].values, expected["S1"].values, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(x, originalX)
    np.testing.assert_array_equal(y, originalY)
