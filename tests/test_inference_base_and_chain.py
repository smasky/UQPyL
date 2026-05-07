import numpy as np
import pytest

from UQPyL.inference.base import InferenceABC
from UQPyL.inference.chain import Chain
from UQPyL.inference.runtime import InfHistory, InfResult
from UQPyL.optimization.base import AlgorithmABC
from UQPyL.problem import ProblemABC
from UQPyL.problem import Eval
from UQPyL.problem.problem import Problem


@ProblemABC.singleEval
def _eval(x):
    # simple objective + always-feasible constraint
    x = np.asarray(x)
    return Eval(objs=float(np.sum(x**2)), cons=np.array([-1.0]))


class DummyInference(InferenceABC):
    name = "DummyInference"

    def run(self):
        raise NotImplementedError


class DummyAlgorithm(AlgorithmABC):
    name = "DummyAlgorithm"

    def run(self, problem, seed=None):
        return None


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
    # cover setup(seed=None) random seed branch through a concrete subclass
    problem = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, evaluate=_eval)
    with pytest.raises(TypeError):
        InferenceABC(maxIters=1, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    inf = DummyInference(maxIters=1, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    inf.setup(problem, seed=None)


def test_inferenceabc_check_gamma_branches():
    problem = Problem(nInput=3, nObj=1, ub=1.0, lb=0.0, evaluate=_eval)
    inf = DummyInference(maxIters=2, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    inf.set("nChains", 4)
    inf.setProblem(problem)

    g = inf._check_gamma_(0.1)
    assert g.shape == (4, 3)

    g2 = inf._check_gamma_(np.array([0.1, 0.2, 0.3]))
    assert g2.shape == (4, 3)

    g3 = inf._check_gamma_(np.ones((4, 3)))
    assert g3.shape == (4, 3)

    with pytest.raises(ValueError):
        inf._check_gamma_(np.ones((2, 3)))  # wrong nChains
    with pytest.raises(ValueError):
        inf._check_gamma_("bad")


def test_inferenceabc_initchains_and_build_result_smoke():
    # include constraint path to cover fixed branches
    problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, evaluate=_eval)
    inf = DummyInference(maxIters=2, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    inf.set("nChains", 2)
    inf.setup(problem, seed=123)

    X0 = np.array([[0.1, 0.2], [0.3, 0.4]])
    objs0, cons0 = inf.evaluate(X0)
    chains = inf.initChains(2, X0, objs0, cons0)
    # fill up to length==maxIters for genNetCDF to be consistent
    for i, c in enumerate(chains):
        c.add(X0[i], objs0[i], cons0[i])

    inf.update(chains)
    res = inf.buildResult()
    assert res.decs.shape == (2, 2, 2)
    assert res.cons.shape == (2, 2, 1)
    assert res.bestDecs.shape == (1, 2)


def test_setup_uses_instance_rng_without_mutating_global_state():
    problem = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, evaluate=_eval)

    np.random.seed(2024)
    expected_after_alg = np.random.RandomState(2024).rand()
    alg = DummyAlgorithm(maxFEs=1, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=123)
    assert hasattr(alg, "rng")
    got_after_alg = np.random.rand()
    assert np.isclose(got_after_alg, expected_after_alg)

    np.random.seed(2025)
    expected_after_inf = np.random.RandomState(2025).rand()
    inf = DummyInference(maxIters=1, verboseFlag=False, verboseFreq=10, logFlag=False, saveFlag=False)
    inf.setup(problem, seed=456)
    assert hasattr(inf, "rng")
    got_after_inf = np.random.rand()
    assert np.isclose(got_after_inf, expected_after_inf)


def test_inference_result_keeps_internal_camelcase_and_exports_snake_case():
    result = InfResult(
        runId="inf_001",
        method="MH",
        problemName="Demo",
        nInput=2,
        nOutput=1,
        nCon=0,
        settings={"nChains": 2},
        runtime=0.2,
        createdAt="2026-05-06T00:00:00",
        decs=np.zeros((1, 1, 2)),
        objs=np.zeros((1, 1, 1)),
        cons=None,
        logProb=np.zeros((1, 1)),
        accepted=np.ones((1, 1), dtype=bool),
        feasibleMask=np.ones((1, 1), dtype=bool),
        acceptanceRate=np.ones((1,)),
        bestDecs=np.zeros((1, 2)),
        bestObjs=np.zeros((1, 1)),
        bestCons=None,
        bestFeasible=True,
        FEs=1,
        iters=0,
        history=InfHistory(),
    )

    assert result.runId == "inf_001"
    assert result.problemName == "Demo"

    summary = result.summary()
    assert summary["run_id"] == "inf_001"
    assert summary["problem_name"] == "Demo"


def test_amh_update_covs_keeps_nontrivial_proposal_scale_after_reinit():
    from UQPyL.inference.methods.amh import AMH

    @ProblemABC.singleEval
    def _eval_gaussian(x):
        x = np.asarray(x)
        return Eval(objs=float(0.5 * np.sum(x**2)))

    problem = Problem(nInput=2, nObj=1, ub=4.0, lb=-4.0, evaluate=_eval_gaussian)
    amh = AMH(nChains=3, warmUp=5, maxIterTimes=12, verboseFlag=False, saveFlag=False)
    amh.setup(problem, seed=2024)

    nChains = amh.get("nChains")
    gamma = amh._check_gamma_(0.2)
    sd = 2.38**2 / problem.nInput
    propCovs = [np.diag(((gamma[i] * (problem.ub - problem.lb))**2).ravel()) for i in range(nChains)]

    X_cur, Objs_cur, Cons_cur = amh.initialSampling(problem, nChains)
    for _ in range(amh.get("warmUp")):
        X_star = amh.f_prop(X_cur, amh.get("propDist"), propCovs, problem.ub, problem.lb)
        Objs_star, Cons_star = amh.evaluate(X_star)
        for i in range(nChains):
            if amh.accept(Objs_star[i], Objs_cur[i], None, decStar=X_star[i], decCur=X_cur[i], consCur=None):
                X_cur[i] = X_star[i]
                Objs_cur[i] = Objs_star[i]
                Cons_cur = Cons_star

    chains = amh.initChains(nChains, X_cur, Objs_cur, Cons_cur)
    X_star = amh.f_prop(X_cur, amh.get("propDist"), propCovs, problem.ub, problem.lb)
    Objs_star, Cons_star = amh.evaluate(X_star)
    for i, chain in enumerate(chains):
        accepted = amh.accept(Objs_star[i], Objs_cur[i], None, decStar=X_star[i], decCur=X_cur[i], consCur=None)
        if accepted:
            X_cur[i] = X_star[i]
            Objs_cur[i] = Objs_star[i]
        chain.add(X_cur[i], Objs_cur[i], None, logProb=amh.log_prob(Objs_cur[i], decs=X_cur[i], cons=None), accepted=accepted)

    updated = amh.updateCovs(chains, sd)
    min_diag = min(float(np.min(np.diag(cov))) for cov in updated)

    assert min_diag > 1e-3


