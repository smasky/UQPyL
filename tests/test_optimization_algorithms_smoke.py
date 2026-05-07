from UQPyL.optimization.moea import NSGAII
from UQPyL.optimization.soea import DE, GA
from UQPyL.problem.mop.ZDT import ZDT1
from UQPyL.problem.sop.single_simple_problem import Sphere
import numpy as np


def test_ga_run_smoke_small_budget():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res = alg.run(problem, seed=123)
    assert res.bestObjs.shape == (1, 1)
    assert res.bestDecs.shape == (1, 2)
    assert res.FEs == 18


def test_de_run_smoke_small_budget():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = DE(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res = alg.run(problem, seed=123)
    assert res.bestObjs.shape == (1, 1)
    assert res.bestDecs.shape == (1, 2)
    assert res.FEs == 18


def test_nsga2_run_smoke_small_budget():
    problem = ZDT1(nInput=6, ub=1.0, lb=0.0)
    alg = NSGAII(nPop=8, maxFEs=24, maxIters=2, verboseFlag=False, logFlag=False, saveFlag=False)
    res = alg.run(problem, seed=123)
    assert res.bestObjs.shape[1] == 2
    assert res.bestMetric is not None
    assert res.FEs == 24


def test_ga_run_does_not_mutate_global_random_state():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    np.random.seed(31415)
    expected_next = np.random.RandomState(31415).rand()

    alg = GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.run(problem, seed=123)

    got_next = np.random.rand()
    assert np.isclose(got_next, expected_next)
