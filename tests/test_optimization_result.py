import numpy as np
import pytest

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


def test_opt_result_exports_snake_case_summary_and_dict():
    history = OptHistory(
        iterToFEs=[[0, 2]],
        bestObjHistory=[0.05],
        numBestHistory=[],
        bestMetricHistory=[],
    )
    result = OptResult(
        bestDecs=np.array([[0.1, 0.2]]),
        bestObjs=np.array([[0.05]]),
        bestCons=None,
        bestMetric=None,
        bestFeasible=True,
        appearFEs=2,
        appearIters=0,
        FEs=2,
        iters=0,
        runtime=0.1,
        history=history,
    )

    assert result.bestFeasible is True
    assert result.appearFEs == 2

    summary = result.summary()
    assert summary["best_feasible"] is True
    assert summary["appear_fes"] == 2
    assert summary["appear_iters"] == 0

    payload = result.toDict()
    assert payload["best_feasible"] is True
    assert payload["appear_fes"] == 2
    assert payload["history"]["iter_to_fes"] == [[0, 2]]


@pytest.mark.parametrize("oldObj,oldCons,newObj,newCons,expectedNew", [
    (0, [10], 100, [1], True),   # Lower violation wins despite worse objective.
    (100, [1], 0, [10], False),  # Better objective cannot hide worse violation.
    (100, [1], 0, [1], False),   # Equal infeasible violations retain incumbent.
    (0, [1], 100, [0], True),    # Feasible always beats infeasible.
    (100, [0], 0, [1], False),
    (100, [-1], 0, [0], True),   # Both feasible: compare objectives.
    (0, [0], 100, [-1], False),
    (100, None, 0, None, True),  # Unconstrained comparison remains unchanged.
])
def test_single_history_uses_feasibility_first_comparison(oldObj, oldCons, newObj, newCons, expectedNew):
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda X: X)
    algorithm = _DummyAlg(problem)
    state = Result(algorithm)
    old = Population([[.2]], [[oldObj]], None if oldCons is None else [oldCons])
    new = Population([[.8]], [[newObj]], None if newCons is None else [newCons])
    state.update(old, problem, FEs=1, iters=0, algType="EA")
    state.update(new, problem, FEs=2, iters=1, algType="EA")
    expected = new if expectedNew else old
    np.testing.assert_array_equal(state.bestDecs, expected.decs)
    np.testing.assert_array_equal(state.bestObjs, expected.objs)
    if expected.cons is None:
        assert state.bestCons is None
    else:
        np.testing.assert_array_equal(state.bestCons, expected.cons)
    assert state.bestFeasible == (expected.cons is None or np.all(expected.cons <= 0))
    assert state.appearFEs == (2 if expectedNew else 1)
    assert state.appearIters == (1 if expectedNew else 0)
    assert state.history.improvedHistory == [True, expectedNew]
    result = state.buildResult()
    np.testing.assert_array_equal(result.bestDecs, expected.decs)
    np.testing.assert_array_equal(result.history.bests[-1]["bestObjs"], expected.objs)


def test_single_history_uses_population_constraint_weights():
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda X: X)
    state = Result(_DummyAlg(problem))
    # Unweighted sums favor the old solution (2 versus 3), but weighted
    # violations favor the new solution (3 versus 20).
    old = Population([[.2]], [[0]], [[2, 0]], conWgt=[10, 1])
    new = Population([[.8]], [[100]], [[0, 3]], conWgt=[10, 1])
    state.update(old, problem, FEs=1, iters=0, algType="EA")
    state.update(new, problem, FEs=2, iters=1, algType="EA")
    np.testing.assert_array_equal(state.bestDecs, new.decs)
    assert not state.bestFeasible


def test_single_history_feasibility_matches_zero_weight_constraint_policy():
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda X: X)
    state = Result(_DummyAlg(problem))
    pop = Population([[.2]], [[0]], [[10, -1]], conWgt=[0, 1])
    state.update(pop, problem, FEs=1, iters=0, algType="EA")
    assert state.bestFeasible
