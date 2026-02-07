import numpy as np
import pytest

from UQPyL.analysis import FAST, RBDFAST, RSA, Sobol, MARS
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _nonlinear_obj(x):
    # non-constant objective to avoid zero variance in spectral methods
    x = np.asarray(x)
    return float(np.sum(np.sin(x) + 0.1 * x**2))


def _make_problem(n_input=3):
    return Problem(nInput=n_input, nOutput=1, ub=1.0, lb=0.0, objFunc=_nonlinear_obj, optType="min")


def test_sobol_sample_validation_and_analyze_smoke():
    problem = _make_problem(3)
    sob = Sobol(verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.raises(ValueError):
        sob.sample(problem, N=10)  # must be power of 2

    X = sob.sample(problem, N=8, secondOrder=True, seed=123)
    assert X.shape == ((2 * problem.nInput + 2) * 8, problem.nInput)
    ds = sob.analyze(problem, X, Y=None, secondOrder=True, target="objFunc", index="all")
    assert "S1" in ds.data_vars and "ST" in ds.data_vars and "S2" in ds.data_vars

    X2 = sob.sample(problem, N=8, secondOrder=False, seed=123)
    ds2 = sob.analyze(problem, X2, Y=None, secondOrder=False, target="objFunc", index="all")
    assert "S1" in ds2.data_vars and "ST" in ds2.data_vars


def test_fast_sample_and_analyze_smoke():
    problem = _make_problem(3)
    fast = FAST(verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.raises(ValueError):
        fast.sample(problem, N=10, M=4, seed=1)  # too small

    # must be strictly greater than 4*M^2 for stable frequency allocation
    X = fast.sample(problem, N=65, M=4, seed=123)
    ds = fast.analyze(problem, X, Y=None, target="objFunc", index="all")
    assert "S1" in ds.data_vars and "ST" in ds.data_vars


def test_rbd_fast_sample_and_analyze_smoke():
    problem = _make_problem(3)
    rbd = RBDFAST(verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.raises(ValueError):
        rbd.sample(problem, N=64, M=4, seed=1)  # must be > 4*M^2

    X = rbd.sample(problem, N=65, M=4, seed=123)
    ds = rbd.analyze(problem, X, Y=None, target="objFunc", index="all")
    assert "S1" in ds.data_vars


def test_rsa_sample_and_analyze_smoke():
    problem = _make_problem(3)
    rsa = RSA(verboseFlag=False, logFlag=False, saveFlag=False)

    X = rsa.sample(problem, N=60, seed=123)
    ds = rsa.analyze(problem, X, Y=None, target="objFunc", index="all", nRegion=4)
    assert "S1" in ds.data_vars and "S1_scale" in ds.data_vars


def test_mars_sample_and_analyze_smoke():
    problem = _make_problem(3)
    mars = MARS(verboseFlag=False, logFlag=False, saveFlag=False)

    X = mars.sample(problem, N=40, seed=123)
    ds = mars.analyze(problem, X, Y=None, target="objFunc", index="all")
    assert "S1" in ds.data_vars and "S1_scale" in ds.data_vars


