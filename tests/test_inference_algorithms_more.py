import numpy as np
import pytest

from UQPyL.inference import AMH, DEMC, DREAM_ZS, MH_Gibbs
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _quad_obj(x):
    x = np.asarray(x)
    return float(np.sum(x**2))


def _make_problem(n_input=2):
    return Problem(nInput=n_input, nOutput=1, ub=1.0, lb=-1.0, objFunc=_quad_obj, optType="min")


def test_mh_gibbs_invalid_propdist_raises():
    with pytest.raises(ValueError):
        MH_Gibbs(propDist="bad")


def test_mh_gibbs_run_smoke():
    problem = _make_problem(2)
    alg = MH_Gibbs(nChains=2, warmUp=0, maxIters=3, propDist="gauss", verboseFlag=False, logFlag=False, saveFlag=False)
    res = alg.run(problem, gamma=0.05, seed=123)
    assert "posterior" in res and "stats" in res and "optimization" in res
    assert "lg" in res["posterior"].data_vars


def test_amh_multi_output_rejected_and_run_smoke():
    p_bad = Problem(nInput=2, nOutput=2, ub=1.0, lb=0.0, objFunc=lambda X: np.zeros((np.atleast_2d(X).shape[0], 2)))
    with pytest.raises(ValueError):
        AMH(nChains=2, warmUp=0, maxIterTimes=3, verboseFlag=False, logFlag=False, saveFlag=False).run(p_bad, seed=1)

    problem = _make_problem(2)
    alg = AMH(nChains=2, warmUp=0, maxIterTimes=3, propDist="gauss", verboseFlag=False, logFlag=False, saveFlag=False)
    res = alg.run(problem, gamma=0.05, seed=123)
    assert "posterior" in res and "stats" in res and "optimization" in res


def test_demc_run_smoke_gamma_none_and_float():
    problem = _make_problem(2)
    # DEMC proposal needs >=3 chains (choose j,k != i)
    alg = DEMC(nChains=3, warmUp=0, maxIterTimes=3, verboseFlag=False, logFlag=False, saveFlag=False)
    res = alg.run(problem, gamma=None, seed=123)
    assert "posterior" in res

    alg2 = DEMC(nChains=3, warmUp=0, maxIterTimes=3, verboseFlag=False, logFlag=False, saveFlag=False)
    res2 = alg2.run(problem, gamma=0.05, seed=123)
    assert "posterior" in res2


def test_dream_zs_adaption_and_snooker_edge_case():
    # unit-test helper branches without long run
    problem = _make_problem(2)
    alg = DREAM_ZS(nChains=3, warmUp=0, maxIters=2, ps=1.0, k=1, jitter=0.0, adpInterval=1, archSize=1, nCR=3, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=123)

    # adaption: hit cr_tries>0 and avg==0 branch, and gamma_scale clipping
    pCR = np.ones(3) / 3
    cr_gain = np.zeros(3)
    cr_tries = np.ones(3)  # >0
    ac_local = np.array([10.0, 10.0, 10.0])
    pCR2, cr_gain2, cr_tries2, gamma_scale2 = alg.adaption(pCR, cr_gain, cr_tries, ac_local, gamma_scale=10.0, acTarget=0.25)
    assert np.isclose(pCR2.sum(), 1.0)
    assert 0.3 <= gamma_scale2 <= 3.0

    # snooker_update: force v_norm2==0 by making X_cur[r]==X_cur[s]
    X_cur = np.array([[0.0, 0.0], [0.0, 0.0], [0.1, 0.1]])
    archive = [X_cur[0], X_cur[1], X_cur[2]]
    x_prop, q_ratio = alg.snooker_update(i=2, X_cur=X_cur, archive=archive, gamma=0.1)
    assert np.all(np.isfinite(x_prop))
    assert np.isfinite(q_ratio)


def test_dream_zs_run_smoke_small_iters():
    # cover the main `run` loop with very small settings
    problem = _make_problem(2)
    alg = DREAM_ZS(
        nChains=3,
        warmUp=0,
        maxIters=2,
        ps=1.0,           # always snooker update
        k=1,
        jitter=0.0,
        adpInterval=1,
        archSize=1,
        acTarget=0.25,
        nCR=3,
        verboseFlag=False,
        logFlag=False,
        saveFlag=False,
    )
    res = alg.run(problem, gamma=0.05, seed=123)
    assert "posterior" in res and "stats" in res and "optimization" in res


def test_amh_helpers_update_covs_and_check_alpha():
    from UQPyL.inference.amh import AMH
    from UQPyL.inference.chain import Chain

    problem = _make_problem(3)
    alg = AMH(nChains=2, warmUp=0, maxIterTimes=3, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=123)
    alg.setParaVal("nChains", 2)

    # _check_alpha branches
    assert alg._check_alpha(0.1).shape == (2, 3)
    assert alg._check_alpha(np.array([0.1, 0.2, 0.3])).shape == (2, 3)
    assert alg._check_alpha(np.ones((2, 3))).shape == (2, 3)

    # updateCovs smoke
    chains = [Chain(3, 1, 0, 3), Chain(3, 1, 0, 3)]
    for c in chains:
        c.add(np.zeros(3), np.zeros(1))
        c.add(np.ones(3) * 0.1, np.ones(1) * 0.01)
    covs = alg.updateCovs(chains, sd=1.0)
    assert len(covs) == 2
    assert covs[0].shape == (3, 3)


def test_demc_helpers_check_alpha_and_f_prop():
    from UQPyL.inference.demc import DEMC

    problem = _make_problem(2)
    alg = DEMC(nChains=3, warmUp=0, maxIterTimes=3, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=123)
    alg.setParaVal("nChains", 3)

    # _check_alpha branches
    assert alg._check_alpha(0.1).shape == (3, 2)
    assert alg._check_alpha(np.array([0.1, 0.2])).shape == (3, 2)
    assert alg._check_alpha(np.ones((3, 2))).shape == (3, 2)

    X_cur = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    X_star = alg.f_prop(X_cur, problem.ub, problem.lb, gamma=None)
    assert X_star.shape == X_cur.shape


