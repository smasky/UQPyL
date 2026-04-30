from UQPyL.optimization.moea import NSGAII
from UQPyL.optimization.soea import DE, GA
from UQPyL.problem.mop.ZDT import ZDT1
from UQPyL.problem.sop.single_simple_problem import Sphere


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
