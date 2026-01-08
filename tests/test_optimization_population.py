import numpy as np

from UQPyL.optimization.population import Population
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _quad_obj(x):
    x = np.asarray(x)
    return float(np.sum(x**2))


def test_population_evaluate_and_get_best_single_objective():
    problem = Problem(nInput=2, nOutput=1, ub=1.0, lb=-1.0, objFunc=_quad_obj, optType="min")

    decs = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    pop = Population(decs)
    pop.evaluate(problem)

    assert pop.objs.shape == (3, 1)
    assert np.allclose(pop.objs.ravel(), [0.0, 0.25, 1.0])

    best = pop.getBest(k=1)
    assert best.decs.shape == (1, 2)
    assert np.allclose(best.decs, [[0.0, 0.0]])


