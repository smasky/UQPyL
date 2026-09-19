from types import SimpleNamespace

import numpy as np
import pytest

from UQPyL.surrogate.util.lbfgsb import LBFGSB


def problemWith(objective):
    return SimpleNamespace(lb=np.array([-2.]), ub=np.array([2.]), nInput=1, objFunc=objective)


@pytest.mark.parametrize('success', [True, False])
def testReturnsBestEvaluatedPairAndPreservesRawStatus(monkeypatch, success):
    def minimize(fun, initial, **kwargs):
        work = initial.copy()
        for value in [1., .25, -.5]:
            work[:] = value
            fun(work)
        return SimpleNamespace(x=work, fun=-123., success=success, status=0 if success else 2)

    monkeypatch.setattr('UQPyL.surrogate.util.lbfgsb.minimize', minimize)
    optimizer = LBFGSB()
    point, score = optimizer.run(problemWith(lambda x: float(x[0]**2)), xInit=[1.])
    np.testing.assert_array_equal(point, [.25])
    assert score == .25**2
    assert optimizer.lastResult.success is success
    assert optimizer.lastResult.fun == -123.


def testRealEarlyStopStillReturnsConsistentImprovement():
    objective = lambda x: float((x[0]-.37)**4)
    optimizer = LBFGSB({'maxiter': 1})
    point, value = optimizer.run(problemWith(objective), xInit=[1.5])
    assert value == objective(point)
    assert value <= objective(np.array([1.5]))
    assert not optimizer.lastResult.success


def testNoFiniteCandidateRaisesAndStateResets(monkeypatch):
    def minimize(fun, initial, **kwargs):
        return SimpleNamespace(x=initial, fun=fun(initial), success=False)

    monkeypatch.setattr('UQPyL.surrogate.util.lbfgsb.minimize', minimize)
    optimizer = LBFGSB()
    optimizer.run(problemWith(lambda x: 1.), xInit=[0.])
    with pytest.raises(RuntimeError, match='no finite'):
        optimizer.run(problemWith(lambda x: np.nan), xInit=[0.])
    assert not optimizer.lastResult.success
    with pytest.raises(ValueError, match='finite value'):
        optimizer.run(problemWith(lambda x: 1.), xInit=[np.nan])
    assert optimizer.lastResult is None


@pytest.mark.parametrize('bounds', [([], []), ([2.], [1.]), ([np.nan], [1.]), ([0.], [np.inf])])
def testInvalidBounds(bounds):
    problem = problemWith(lambda x: 1.)
    problem.lb, problem.ub = map(np.array, bounds)
    with pytest.raises(ValueError):
        LBFGSB().run(problem)
