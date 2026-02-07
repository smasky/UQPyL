import numpy as np

from UQPyL.optimization.population import Population
from UQPyL.optimization.result import Result
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
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

        class _S:
            dicts = {"dummy": 1}

        self.setting = _S()


def test_result_update_ea_and_generate_netcdf_smoke():
    problem = Problem(nInput=2, nOutput=1, ub=1.0, lb=-1.0, objFunc=_obj_single, optType="min")
    alg = _DummyAlg(problem)
    res = Result(alg)

    decs = np.array([[0.1, 0.2], [0.9, -0.9]])
    pop = Population(decs)
    pop.evaluate(problem)

    res.update(pop, problem, FEs=2, iter=0, algType="EA")
    out = res.generateNetCDF()
    assert "history" in out and "result" in out
    assert "bestDecs" in out["result"].data_vars


def test_result_update_moea_and_generate_netcdf_smoke():
    problem = Problem(nInput=2, nOutput=2, ub=1.0, lb=0.0, objFunc=_obj_multi, optType="min")
    alg = _DummyAlg(problem)
    res = Result(alg)

    decs = np.array([[0.1, 0.2], [0.9, 0.1], [0.5, 0.5]])
    pop = Population(decs)
    pop.evaluate(problem)

    res.update(pop, problem, FEs=3, iter=0, algType="MOEA")
    out = res.generateNetCDF()
    assert "history" in out and "result" in out
    assert "bestMetric" in out["result"].data_vars

