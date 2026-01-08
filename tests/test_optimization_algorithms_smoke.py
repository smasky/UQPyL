from UQPyL.optimization.moea import NSGAII
from UQPyL.optimization.soea import DE, GA
from UQPyL.problem.mop.ZDT import ZDT1
from UQPyL.problem.sop.single_simple_problem import Sphere


def test_ga_run_smoke_small_budget():
    # Use built-in benchmark problem
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    assert "history" in res_nc and "result" in res_nc
    assert "bestObjs" in res_nc["result"].data_vars


def test_de_run_smoke_small_budget():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = DE(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    assert "history" in res_nc and "result" in res_nc
    assert "bestObjs" in res_nc["result"].data_vars


def test_nsga2_run_smoke_small_budget():
    problem = ZDT1(nInput=6, ub=1.0, lb=0.0)
    alg = NSGAII(nPop=8, maxFEs=24, maxIters=2, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    assert "history" in res_nc and "result" in res_nc
    assert "bestObjs" in res_nc["result"].data_vars
    assert "bestMetric" in res_nc["result"].data_vars

