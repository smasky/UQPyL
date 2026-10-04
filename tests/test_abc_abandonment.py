import numpy as np
import pytest

from UQPyL.optimization.soea import ABC
from UQPyL.optimization.population import Population
from UQPyL.problem import Problem


QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


@pytest.mark.parametrize("employedRate", [0.25, 0.75])
def testAbandonedBeesAreReevaluatedAndCountersReset(employedRate):
    received = []

    def objective(X):
        received.append(X.copy())
        return np.sum(X**2, axis=1, keepdims=True)

    problem = Problem(nInput=2, nObj=1, lb=[10, -4], ub=[20, -2], objFunc=objective)
    method = ABC(nPop=4, **QUIET)
    method.setup(problem, seed=9)
    pop = Population(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]))
    method.evaluate(pop)
    original = pop.decs.copy()
    beeType = np.array([1, 2, 0, 2])
    counts = np.array([1.0, 5.0, 2.0, 6.0])
    pop, beeType, counts = method.updateOnlookerBees(pop, beeType, counts, employedRate)
    assert method.FEs == sum(len(batch) for batch in received) == 6
    np.testing.assert_array_equal(pop.decs[[0, 2]], original[[0, 2]])
    assert np.all((pop.decs >= 0) & (pop.decs <= 1))
    assert not np.array_equal(pop.decs[[1, 3]], original[[1, 3]])
    np.testing.assert_array_equal(counts, [1, 0, 2, 0])
    np.testing.assert_array_equal(beeType, [1, 0, 0, 0] if employedRate == 0.25 else [1, 1, 0, 1])
    realX = problem.unit_to_space(pop.decs)
    np.testing.assert_allclose(pop.objs, np.sum(realX**2, axis=1, keepdims=True))


def testAbandonmentUsesStrictLimitBoundary():
    method = ABC(**QUIET)
    np.testing.assert_array_equal(method.checkLimitTimes(np.array([1, 1, 0]), np.array([2, 3, 4]), 3), [1, 1, 2])


def testShortRunActuallyResetsStagnantBees(monkeypatch):
    evaluated = []
    resets = []

    def objective(X):
        evaluated.append(X.copy())
        return np.ones((len(X), 1))

    method = ABC(nPop=8, employedRate=0.5, limit=0, maxIters=4, **QUIET)
    originalUpdate = method.updateOnlookerBees

    def recordReset(pop, beeType, counts, rate):
        abandoned = np.flatnonzero(beeType == 2)
        previousFEs = method.FEs
        result = originalUpdate(pop, beeType, counts, rate)
        if len(abandoned):
            resets.append(len(abandoned))
            assert method.FEs - previousFEs == len(abandoned)
            np.testing.assert_array_equal(result[2][abandoned], 0)
        return result

    monkeypatch.setattr(method, "updateOnlookerBees", recordReset)
    result = method.run(Problem(nInput=2, nObj=1, lb=-3, ub=7, objFunc=objective), seed=13)
    assert resets
    assert result.FEs == sum(map(len, evaluated))
    assert np.all((np.vstack(evaluated) >= -3) & (np.vstack(evaluated) <= 7))
