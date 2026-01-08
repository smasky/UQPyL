import numpy as np
import pytest

from UQPyL.inference.base import InferenceABC
from UQPyL.inference.chain import Chain
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleEval
def _eval(x):
    # simple objective + always-feasible constraint
    x = np.asarray(x)
    return {"objs": float(np.sum(x**2)), "cons": np.array([-1.0])}


class DummyInference(InferenceABC):
    name = "DummyInference"

    def run(self):
        raise NotImplementedError


def test_chain_add_and_count():
    c = Chain(nInput=2, nOutput=1, nCons=0, length=3)
    assert c.count == 0
    c.add(np.array([0.0, 1.0]), np.array([2.0]))
    assert c.count == 1
    assert np.allclose(c.decs[0], [0.0, 1.0])
    assert np.allclose(c.objs[0], [2.0])
    assert c.cons is None


def test_chain_with_constraints_stores_cons():
    c = Chain(nInput=2, nOutput=1, nCons=1, length=2)
    c.add(np.array([0.0, 0.0]), np.array([0.0]), np.array([-1.0]))
    assert c.cons is not None
    assert np.allclose(c.cons[0], [-1.0])


def test_inferenceabc_check_bound_reflection():
    inf = DummyInference(maxIters=2, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    ub = np.array([[1.0, 1.0]])
    lb = np.array([[0.0, 0.0]])
    X = np.array([[1.2, -0.2], [2.2, -1.2]])
    Xr = inf._check_bound_(X, ub, lb)
    assert np.all(Xr >= lb - 1e-12)
    assert np.all(Xr <= ub + 1e-12)


def test_inferenceabc_run_placeholder_and_setup_seed_none_branch():
    # cover base.run (pass) and setup(seed=None) random seed branch
    problem = Problem(nInput=2, nOutput=1, ub=1.0, lb=0.0, evaluate=_eval)
    inf = InferenceABC(maxIters=1, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    assert inf.run() is None
    inf.setup(problem, seed=None)


def test_inferenceabc_check_gamma_branches():
    problem = Problem(nInput=3, nOutput=1, ub=1.0, lb=0.0, evaluate=_eval)
    inf = DummyInference(maxIters=2, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    inf.setParaVal("nChains", 4)
    inf.setProblem(problem)

    g = inf._check_gamma_(0.1)
    assert g.shape == (4 * 3,)

    g2 = inf._check_gamma_(np.array([0.1, 0.2, 0.3]))
    assert g2.shape == (4 * 3,)

    g3 = inf._check_gamma_(np.ones((4, 3)))
    assert g3.shape == (4 * 3,)

    with pytest.raises(ValueError):
        inf._check_gamma_(np.ones((2, 3)))  # wrong nChains
    with pytest.raises(ValueError):
        inf._check_gamma_("bad")


def test_inferenceabc_initchains_generateverb_and_gennetcdf_smoke():
    # include constraint path to cover fixed branches
    problem = Problem(nInput=2, nOutput=1, ub=1.0, lb=0.0, evaluate=_eval)
    # `Problem` wrapper doesn't expose nCons in constructor; set it explicitly for inference storage branches.
    problem.nCons = 1
    inf = DummyInference(maxIters=2, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    inf.setParaVal("nChains", 2)
    inf.setup(problem, seed=123)

    X0 = np.array([[0.1, 0.2], [0.3, 0.4]])
    objs0, cons0 = inf.evaluate(X0)
    chains = inf.initChains(2, X0, objs0, cons0)
    # fill up to length==maxIters for genNetCDF to be consistent
    for i, c in enumerate(chains):
        c.add(X0[i], objs0[i], cons0[i])

    inf.iter = 1
    verb = inf.generateVerb(chains)
    assert "iter" in verb and "bestObjs" in verb and "bestDecs" in verb

    res = inf.genNetCDF(chains, problem)
    assert set(res.keys()) >= {"posterior", "stats", "optimization"}
    assert "decs" in res["posterior"].data_vars
    assert "cons" in res["posterior"].data_vars


