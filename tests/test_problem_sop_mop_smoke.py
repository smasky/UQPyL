import numpy as np

from UQPyL.problem.mop import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from UQPyL.problem.sop import (
    Ackley,
    Bent_Cigar,
    Discus,
    Griewank,
    Quartic,
    Rastrigin,
    Rosenbrock,
    RosenbrockWithCon,
    Schwefel_1_22,
    Schwefel_2_21,
    Schwefel_2_22,
    Schwefel_2_26,
    Sphere,
    Step,
    Trid,
    Weierstrass,
)


def _smoke_single(cls, n_input=5, seed=123):
    p = cls(nInput=n_input)
    rng = np.random.default_rng(seed)
    X = rng.uniform(p.lb, p.ub, size=(4, p.nInput))
    Y = p.objFunc(X)
    assert Y.shape == (4, p.nOutput)
    assert np.isfinite(Y).all()
    return p


def test_sop_all_obj_funcs_smoke():
    # keep n_input small so it runs fast
    for cls in [
        Sphere,
        Schwefel_2_22,
        Schwefel_1_22,
        Schwefel_2_21,
        Rosenbrock,
        Step,
        Quartic,
        Schwefel_2_26,
        Rastrigin,
        Ackley,
        Griewank,
        Trid,
        Bent_Cigar,
        Discus,
        Weierstrass,
    ]:
        _smoke_single(cls, n_input=5)


def test_sop_constraint_problem_smoke():
    p = RosenbrockWithCon(nInput=3)
    rng = np.random.default_rng(123)
    X = rng.uniform(p.lb, p.ub, size=(4, p.nInput))
    Y = p.objFunc(X)
    C = p.conFunc(X)
    assert Y.shape == (4, 1)
    assert C.shape == (4, 1)


def test_mop_zdt_smoke_and_optimum():
    rng = np.random.default_rng(123)
    for cls in [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]:
        p = cls(nInput=10, nOutput=2)
        X = rng.uniform(p.lb, p.ub, size=(4, p.nInput))
        Y = p.objFunc(X)
        assert Y.shape == (4, 2)
        assert np.isfinite(Y).all()
        opt = p.getOptimum(N=10) if "N" in p.getOptimum.__code__.co_varnames else p.getOptimum(10)
        assert opt.shape[1] == 2


def test_mop_dtlz_smoke_and_optimum():
    rng = np.random.default_rng(123)

    # DTLZ1: any nOutput default 3; keep small input
    p1 = DTLZ1(nInput=7, nOutput=3)
    X = rng.uniform(p1.lb, p1.ub, size=(4, p1.nInput))
    Y = p1.objFunc(X)
    assert Y.shape == (4, 3)
    opt = p1.getOptimum(10)
    assert opt.shape[1] == 3

    # DTLZ2 requires 3 objectives in this implementation
    p2 = DTLZ2(nInput=7, nOutput=3)
    Y2 = p2.objFunc(rng.uniform(p2.lb, p2.ub, size=(4, p2.nInput)))
    assert Y2.shape == (4, 3)
    assert p2.getOptimum(10).shape[1] == 3

    for cls in [DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7]:
        p = cls(nInput=7, nOutput=3)
        Y = p.objFunc(rng.uniform(p.lb, p.ub, size=(4, p.nInput)))
        assert Y.shape == (4, p.nOutput)
        assert np.isfinite(Y).all()
        opt = p.getOptimum(10)
        assert opt.shape[1] == p.nOutput


