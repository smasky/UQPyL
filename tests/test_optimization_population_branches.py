import numpy as np

from UQPyL.optimization.population import Population


def test_population_argsort_single_with_constraints_branch():
    decs = np.array([[0.0], [1.0], [2.0]])
    objs = np.array([[1.0], [0.0], [2.0]])
    cons = np.array([[-1.0], [1.0], [0.0]])  # one infeasible
    pop = Population(decs, objs, cons=cons, conWgt=np.array([[1.0]]))
    order = pop.argsort()
    assert order.shape == (3,)


def test_population_best_multi_feasible_and_crowding_k_branch():
    decs = np.array([[0.0, 0.0], [0.2, 0.8], [0.8, 0.2], [1.0, 1.0]])
    objs = np.array([[0.0, 2.0], [0.5, 1.5], [1.5, 0.5], [2.0, 0.0]])
    cons = -np.ones((4, 1))  # all feasible
    pop = Population(decs, objs, cons=cons)
    best = pop.getBest(k=2)  # forces crowding distance truncation branch
    assert len(best) == 2


def test_population_best_multi_infeasible_only_branch_returns_topk_by_cv():
    decs = np.array([[0.0, 0.0], [0.2, 0.8], [0.8, 0.2], [1.0, 1.0]])
    objs = np.array([[0.0, 2.0], [0.5, 1.5], [1.5, 0.5], [2.0, 0.0]])
    cons = np.array([[1.0], [2.0], [0.5], [3.0]])  # all infeasible
    pop = Population(decs, objs, cons=cons)
    best = pop.getBest(k=2)
    assert len(best) == 2


def test_population_get_pareto_front_constraint_branches():
    decs = np.array([[0.0, 0.0], [0.2, 0.8], [0.8, 0.2], [1.0, 1.0]])
    objs = np.array([[0.0, 2.0], [0.5, 1.5], [1.5, 0.5], [2.0, 0.0]])
    cons = np.array([[-1.0], [1.0], [-1.0], [2.0]])  # mix feasible/infeasible
    pop = Population(decs, objs, cons=cons)
    pf = pop.getParetoFront()
    assert len(pf) >= 1


def test_population_clip_replace_size_merge():
    pop = Population(np.array([[2.0, -2.0], [0.0, 0.0]]), np.array([[2.0], [0.0]]))
    pop.clip(lb=-1.0, ub=1.0)
    assert np.all(pop.decs <= 1.0) and np.all(pop.decs >= -1.0)

    other = Population(np.array([[0.1, 0.1]]), np.array([[0.01]]))
    pop.replace(np.array([True, False]), other)
    assert np.allclose(pop.decs[0], [0.1, 0.1])

    n, d = pop.size()
    assert (n, d) == pop.decs.shape

    merged = pop.merge(Population(np.array([[0.2, 0.2]]), np.array([[0.04]])))
    assert isinstance(merged, Population)
    assert len(merged) == 3

