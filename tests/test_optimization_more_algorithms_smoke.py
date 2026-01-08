import numpy as np

from UQPyL.optimization.moea import MOEAD, NSGAIII, RVEA, MOASMO
from UQPyL.optimization.soea import ABC, CSA, EGO, ASMO
from UQPyL.problem.mop.ZDT import ZDT1
from UQPyL.problem.sop.single_simple_problem import Sphere


def _assert_netcdf_dict(res_nc):
    assert isinstance(res_nc, dict)
    assert "history" in res_nc and "result" in res_nc
    assert "bestDecs" in res_nc["result"].data_vars
    assert "bestObjs" in res_nc["result"].data_vars


class _DummySurrogate:
    def fit(self, X, Y):
        return self

    def predict(self, X, only_value=True):
        X = np.atleast_2d(X)
        y = np.sum(X**2, axis=1, keepdims=True)
        if only_value:
            return y
        mse = np.ones_like(y) * 0.1
        return y, mse


class _DummyMultiSurrogate:
    def __init__(self, n_out):
        self.n_out = n_out

    def fit(self, X, Y):
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        # return (n, n_out)
        base = np.sum(X**2, axis=1, keepdims=True)
        return np.hstack([base + 0.1 * i for i in range(self.n_out)])


def test_abc_runs_on_sphere_small_budget():
    problem = Sphere(nInput=3, ub=1.0, lb=-1.0)
    alg = ABC(nPop=10, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)


def test_csa_runs_on_sphere_small_budget():
    problem = Sphere(nInput=3, ub=1.0, lb=-1.0)
    alg = CSA(nPop=10, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)


def test_ego_runs_on_sphere_with_dummy_surrogate_and_small_inner_ga():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = EGO(nInit=6, maxFEs=20, maxIters=3, verboseFlag=False, logFlag=False, saveFlag=False)
    # make surrogate fast/stable
    alg.surrogate = _DummySurrogate()
    # shrink inner GA budget
    from UQPyL.optimization.soea import GA as _GA
    alg.optimizer = _GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)


def test_asmo_one_step_runs_on_sphere_with_dummy_surrogate():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    alg = ASMO(nInit=6, maxFEs=20, maxIters=3, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.surrogate = _DummySurrogate()
    res_nc = alg.run(problem, seed=123, oneStep=True)
    _assert_netcdf_dict(res_nc)


def test_moead_runs_on_zdt1_small_budget():
    problem = ZDT1(nInput=6, ub=1.0, lb=0.0)
    alg = MOEAD(nPop=12, maxFEs=40, maxIters=3, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)


def test_nsgaiii_runs_on_zdt1_small_budget():
    problem = ZDT1(nInput=6, ub=1.0, lb=0.0)
    alg = NSGAIII(nPop=12, maxFEs=40, maxIters=3, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)


def test_rvea_runs_on_zdt1_small_budget():
    problem = ZDT1(nInput=6, ub=1.0, lb=0.0)
    alg = RVEA(nPop=12, maxFEs=40, maxIters=3, verboseFlag=False, logFlag=False, saveFlag=False)
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)


def test_moasmo_runs_on_zdt1_with_dummy_multisurrogate_small_budget():
    problem = ZDT1(nInput=6, ub=1.0, lb=0.0)
    alg = MOASMO(
        surrogates=_DummyMultiSurrogate(problem.nOutput),
        optimizer=NSGAIII(nPop=12, maxFEs=40, maxIters=2, verboseFlag=False, logFlag=False, saveFlag=False),
        pct=0.5,
        nInit=6,
        maxFEs=20,
        maxIters=2,
        verboseFlag=False,
        logFlag=False,
        saveFlag=False,
    )
    res_nc = alg.run(problem, seed=123)
    _assert_netcdf_dict(res_nc)

