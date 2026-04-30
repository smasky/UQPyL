import numpy as np

from UQPyL.optimization.population import Population
from UQPyL.optimization.runtime import OptHistory, OptResult, Result
from UQPyL.problem import ProblemBase
from UQPyL.problem.problem import Problem


@ProblemBase.singleFunc
def _obj_single(x):
    x = np.asarray(x)
    return float(np.sum(x**2))


def _obj_multi(X):
    X = np.atleast_2d(X)
    f1 = np.sum(X**2, axis=1)
    f2 = np.sum((X - 0.5) ** 2, axis=1)
    return np.vstack([f1, f2]).T


class _DummyAlg:
    def __init__(self, problem):
        self.name = "DummyAlg"
        self.problem = problem
        self.iters = 0
        self.FEs = 0


def test_result_update_single_builds_opt_result():
    problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=_obj_single, optType="min")
    alg = _DummyAlg(problem)
    state = Result(alg)

    decs = np.array([[0.1, 0.2], [0.9, -0.9]])
    objs = problem.objFunc(decs)
    pop = Population(decs, objs=objs)

    state.update(pop, problem, FEs=2, iters=0, algType="EA")
    alg.FEs = 2
    alg.iters = 0
    result = state.buildResult()

    assert isinstance(state.history, OptHistory)
    assert isinstance(result, OptResult)
    assert result.bestObjs.shape == (1, 1)
    assert result.bestDecs.shape == (1, 2)
    assert len(result.history.bestObjHistory) == 1


def test_result_update_multi_builds_opt_result():
    problem = Problem(nInput=2, nObj=2, ub=1.0, lb=0.0, objFunc=_obj_multi, optType="min")
    alg = _DummyAlg(problem)
    state = Result(alg)

    decs = np.array([[0.1, 0.2], [0.9, 0.1], [0.5, 0.5]])
    objs = problem.objFunc(decs)
    pop = Population(decs, objs=objs)

    state.update(pop, problem, FEs=3, iters=0, algType="MOEA")
    alg.FEs = 3
    alg.iters = 0
    result = state.buildResult()

    assert isinstance(result, OptResult)
    assert result.bestObjs.shape[1] == 2
    assert result.bestMetric is not None
    assert len(result.history.bestMetricHistory) == 1
    assert len(result.history.numBestHistory) == 1


def test_result_multi_freezes_auto_generated_hv_reference_point():
    problem = Problem(nInput=2, nObj=2, ub=1.0, lb=0.0, objFunc=_obj_multi, optType="min")
    alg = _DummyAlg(problem)
    state = Result(alg)

    pop1 = Population(
        decs=np.array([[0.1, 0.2], [0.8, 0.3]]),
        objs=np.array([[0.2, 0.9], [0.8, 0.3]]),
    )
    state.update(pop1, problem, FEs=2, iters=0, algType="MOEA")

    ref1 = state.hvRefPoint.copy()
    expected = np.array([0.96, 1.08])
    assert np.allclose(ref1, expected)

    pop2 = Population(
        decs=np.array([[0.2, 0.2], [0.9, 0.1]]),
        objs=np.array([[0.1, 1.5], [1.2, 0.2]]),
    )
    state.update(pop2, problem, FEs=4, iters=1, algType="MOEA")

    assert np.allclose(state.hvRefPoint, ref1)


def test_result_multi_prefers_explicit_hv_reference_point():
    problem = Problem(nInput=2, nObj=2, ub=1.0, lb=0.0, objFunc=_obj_multi, optType="min")
    alg = _DummyAlg(problem)
    alg.hvRefPoint = np.array([5.0, 6.0])
    state = Result(alg)

    pop = Population(
        decs=np.array([[0.1, 0.2], [0.8, 0.3]]),
        objs=np.array([[0.2, 0.9], [0.8, 0.3]]),
    )
    state.update(pop, problem, FEs=2, iters=0, algType="MOEA")

    assert np.allclose(state.hvRefPoint, np.array([5.0, 6.0]))
