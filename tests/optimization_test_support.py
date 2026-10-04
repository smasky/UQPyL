"""Fresh optimizer factories shared by stopping and persistence tests."""

from UQPyL.optimization.soea import GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.optimization.expensive import EGO, ASMO, MOASMO

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)

METHODS = [GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA, NSGAII, NSGAIII, MOEAD, RVEA, EGO, ASMO, MOASMO]

MULTI = [NSGAII, NSGAIII, MOEAD, RVEA, MOASMO]


def makeMethod(cls, limit):
    options = dict(maxIters=limit, maxFEs=1000, historyFreq=1, **QUIET)
    if cls in [SCE_UA, ML_SCE_UA]:
        return cls(ngs=2, tolerate=None, **options)
    if cls in [EGO, ASMO]:
        method = cls(nInit=6, **options)
        method.optimizer = GA(nPop=6, maxFEs=12, tolerate=None, **QUIET)
        return method
    if cls is MOASMO:
        return cls(nInit=8, pct=0.25, optimizer=NSGAII(nPop=8, maxFEs=16, **QUIET), **options)
    return cls(nPop=12, tolerate=None, **options)


def assertBenchmarkResult(result, problem):
    """Check stored Sphere/ZDT1 objectives against independent formulas."""
    import numpy as np
    from UQPyL.optimization.runtime import OptResult
    from UQPyL.problem.mop.ZDT import ZDT1
    from UQPyL.problem.sop.single_simple_problem import Sphere

    assert isinstance(result, OptResult)
    assert result.bestDecs is not None
    assert result.bestObjs is not None
    assert result.history is not None
    points = np.atleast_2d(result.bestDecs)
    assert points.shape[1] == problem.nInput
    assert np.all(np.isfinite(points))
    assert np.all(points >= problem.lb) and np.all(points <= problem.ub)
    if isinstance(problem, Sphere):
        expected = np.sum(points**2, axis=1, keepdims=True)
    elif isinstance(problem, ZDT1):
        g = 1 + 9 * np.mean(points[:, 1:], axis=1)
        expected = np.column_stack((points[:, 0], g * (1 - np.sqrt(points[:, 0] / g))))
    else:
        raise AssertionError("An independent benchmark formula is required.")
    np.testing.assert_allclose(result.bestObjs, expected, rtol=1e-12, atol=1e-12)
