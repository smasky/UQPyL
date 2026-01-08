import numpy as np

from UQPyL.optimization.base import AlgorithmABC
from UQPyL.optimization.util.ga_operator import gaOperator
from UQPyL.optimization.util.ga_operator import gaOperatorHalf
from UQPyL.optimization.util.uniform_point import uniformPoint
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _obj_single(x):
    x = np.asarray(x)
    return float(np.sum(x**2))


class _DummyAlg(AlgorithmABC):
    name = "DummyAlg"
    alg_type = "EA"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def test_algorithmabc_setup_seed_none_and_save_result_branches(monkeypatch):
    problem = Problem(nInput=2, nOutput=1, ub=1.0, lb=-1.0, objFunc=_obj_single, optType="min")
    alg = _DummyAlg(maxFEs=1, maxIters=1, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=None)  # seed None branch

    called = {"ea": 0, "moea": 0}

    def _save(**kwargs):
        if kwargs.get("alg_type", 0) == 1:
            called["moea"] += 1
        else:
            called["ea"] += 1

    monkeypatch.setattr(alg.result, "save", _save, raising=False)
    alg.saveResult()
    assert called["ea"] == 1

    # multi-objective branch
    problem2 = Problem(nInput=2, nOutput=2, ub=1.0, lb=0.0, objFunc=lambda X: np.zeros((np.atleast_2d(X).shape[0], 2)))
    alg.setup(problem2, seed=123)
    alg.saveResult()
    assert called["moea"] == 1


def test_ga_operator_smoke_and_bounds():
    rng = np.random.default_rng(123)
    decs = rng.random((6, 2))
    ub = np.ones((1, 2))
    lb = np.zeros((1, 2))
    off = gaOperator(decs, ub, lb, proC=1.0, proM=1.0)
    assert off.shape == (6, 2)
    assert np.all(off >= lb - 1e-12)
    assert np.all(off <= ub + 1e-12)


def test_ga_operator_half_smoke_and_bounds():
    rng = np.random.default_rng(123)
    decs = rng.random((6, 2))
    ub = np.ones((1, 2))
    lb = np.zeros((1, 2))
    off = gaOperatorHalf(decs, ub, lb, proC=1.0, disC=20, proM=1.0, disM=20)
    assert off.shape[1] == 2
    assert np.all(off >= lb - 1e-12)
    assert np.all(off <= ub + 1e-12)


def test_uniform_point_nbi_h2_branch():
    # N=9, M=3 -> triggers H1 < M and H2 > 0 branch (lines 32-39 in report)
    W, N = uniformPoint(9, 3, method="NBI")
    assert W.shape[1] == 3
    assert N == W.shape[0]
    assert np.allclose(W.sum(axis=1), 1.0)

