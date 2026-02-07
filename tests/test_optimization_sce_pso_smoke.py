from UQPyL.optimization.soea import PSO, SCE_UA, ML_SCE_UA
from UQPyL.problem.sop.single_simple_problem import Sphere


def _assert_netcdf_dict(res_nc):
    assert isinstance(res_nc, dict)
    assert "history" in res_nc and "result" in res_nc
    assert "bestObjs" in res_nc["result"].data_vars


def test_sce_ua_runs_on_sphere_small_budget():
    # ngs=0 triggers internal branch: ngs <- nInput (keeps runtime low)
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = SCE_UA(ngs=0, maxFEs=25, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)


def test_ml_sce_ua_runs_on_sphere_small_budget():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = ML_SCE_UA(ngs=0, maxFEs=25, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)


def test_pso_runs_on_sphere_and_hits_random_particle_branch():
    # Use nInput=5 and nPop=10 so _randomParticle reinitializes at least 1 element.
    problem = Sphere(nInput=5, ub=1.0, lb=-1.0)
    alg = PSO(nPop=10, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)

