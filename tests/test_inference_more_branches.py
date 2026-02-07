import numpy as np
import pytest

from UQPyL.inference import AMH, DEMC, MH, MH_Gibbs, DREAM_ZS
from UQPyL.inference.base import InferenceABC
from UQPyL.problem import ProblemABC


class ConstrainedQuadratic(ProblemABC):
    """Minimal constrained single-objective problem for inference branch coverage."""

    name = "ConstrainedQuadratic"

    def __init__(self, nInput=2):
        super().__init__(nInput=nInput, nOutput=1, ub=1.0, lb=-1.0, nCons=1, optType="min")

    def objFunc(self, X):
        X = self._check_X_2d(X)
        return np.sum(X**2, axis=1, keepdims=True)

    def conFunc(self, X):
        X = self._check_X_2d(X)
        # always feasible (<=0)
        return -np.ones((X.shape[0], 1))


def test_inferenceabc_checktermination_verbose_branch(monkeypatch):
    # Cover InferenceABC.checkTermination verbose branch (lines 116-118 in report)
    p = ConstrainedQuadratic(nInput=2)
    inf = InferenceABC(maxIters=2, verboseFlag=True, verboseFreq=1, logFlag=False, saveFlag=False)
    inf.setup(p, seed=123)
    inf.setParaVal("nChains", 1)

    # dummy chain list
    X0 = np.array([[0.0, 0.0]])
    objs0, cons0 = inf.evaluate(X0)
    chains = inf.initChains(1, X0, objs0, cons0)

    # silence verbose output
    from UQPyL.util import Verbose as VerboseMod

    monkeypatch.setattr(VerboseMod, "verboseInference", lambda *args, **kwargs: None)
    assert inf.checkTermination(chains) is True


def test_inference_setting_getval_tuple_branch():
    inf = InferenceABC(maxIters=1, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    inf.setParaVal("a", 1)
    inf.setParaVal("b", 2)
    assert inf.getParaVal("a", "b") == (1, 2)


def test_mh_invalid_propdist_raises():
    with pytest.raises(ValueError):
        MH(propDist="bad")


def test_mh_uniform_propdist_smoke_with_constraints():
    p = ConstrainedQuadratic(nInput=2)
    alg = MH(nChains=2, warmUp=1, maxIters=3, propDist="uniform", verboseFlag=False, logFlag=False, saveFlag=False)
    res = alg.run(p, gamma=0.05, seed=123)
    assert "posterior" in res


def test_mh_gibbs_uniform_propdist_smoke_with_constraints():
    p = ConstrainedQuadratic(nInput=2)
    alg = MH_Gibbs(nChains=2, warmUp=1, maxIters=3, propDist="uniform", verboseFlag=False, logFlag=False, saveFlag=False)
    res = alg.run(p, gamma=0.05, seed=123)
    assert "posterior" in res


def test_amh_invalid_propdist_raises():
    with pytest.raises(ValueError):
        AMH(propDist="bad")


def test_amh_propdist_else_branch_raises_direct_call():
    # cover the else branch inside AMH.f_prop (invalid propDist)
    p = ConstrainedQuadratic(nInput=2)
    alg = AMH(nChains=2, warmUp=0, maxIterTimes=2, propDist="gauss", verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(p, seed=123)
    X_cur = np.zeros((2, 2))
    covs = [np.eye(2), np.eye(2)]
    with pytest.raises(ValueError):
        alg.f_prop(X_cur, "bad", covs, p.ub, p.lb)


def test_demc_warmup_branch_and_check_alpha_error_branch():
    p = ConstrainedQuadratic(nInput=2)
    alg = DEMC(nChains=3, warmUp=1, maxIterTimes=3, verboseFlag=False, logFlag=False, saveFlag=False)
    res = alg.run(p, gamma=0.05, seed=123)
    assert "posterior" in res

    # _check_alpha error branches
    alg.setup(p, seed=123)
    alg.setParaVal("nChains", 3)
    with pytest.raises(ValueError):
        alg._check_alpha(np.ones((2, 2)))
    with pytest.raises(ValueError):
        alg._check_alpha("bad")


def test_dream_zs_gamma_none_and_de_prop_path_smoke():
    # run a tiny iteration to cover gamma None branch and de_prop path (ps=0)
    p = ConstrainedQuadratic(nInput=2)
    alg = DREAM_ZS(
        nChains=4,
        warmUp=1,
        ps=0.0,  # always DE proposal, sets crIdxs
        k=1,
        jitter=0.0,
        adpInterval=1,
        archSize=1,
        acTarget=0.25,
        nCR=3,
        maxIters=2,
        verboseFlag=False,
        logFlag=False,
        saveFlag=False,
    )
    res = alg.run(p, gamma=None, seed=123)
    assert "posterior" in res


