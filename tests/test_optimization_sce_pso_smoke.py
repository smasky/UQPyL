import numpy as np

from UQPyL.optimization.runtime import OptResult
from UQPyL.optimization.soea import ML_SCE_UA, PSO, SCE_UA
from UQPyL.problem.sop.single_simple_problem import Sphere


def _assert_opt_result(result):
    assert isinstance(result, OptResult)
    assert result.bestDecs is not None
    assert result.bestObjs is not None
    assert result.history is not None


def test_sce_ua_runs_on_sphere_small_budget():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = SCE_UA(ngs=0, maxFEs=25, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    result = alg.run(problem, seed=123)
    _assert_opt_result(result)


def test_ml_sce_ua_runs_on_sphere_small_budget():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = ML_SCE_UA(ngs=0, maxFEs=25, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    result = alg.run(problem, seed=123)
    _assert_opt_result(result)


def test_pso_runs_on_sphere_and_hits_random_particle_branch():
    problem = Sphere(nInput=5, ub=1.0, lb=-1.0)
    alg = PSO(nPop=10, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    result = alg.run(problem, seed=123)
    _assert_opt_result(result)


def test_pso_run_does_not_mutate_global_random_state():
    problem = Sphere(nInput=5, ub=1.0, lb=-1.0)
    np.random.seed(27182)
    expected_next = np.random.RandomState(27182).rand()

    alg = PSO(nPop=10, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.run(problem, seed=123)

    got_next = np.random.rand()
    assert np.isclose(got_next, expected_next)
