import numpy as np
from pathlib import Path

from UQPyL.inference import AMH, DEMC, DREAM_ZS, InfReader, MH, MH_Gibbs
from UQPyL.problem import Problem


def _gaussian_energy(X):
    X = np.asarray(X)
    return 0.5 * np.sum(X**2, axis=1, keepdims=True)


def _small_gaussian_problem(nInput=2):
    return Problem(
        nInput=nInput,
        nObj=1,
        ub=np.ones(nInput) * 4.0,
        lb=np.ones(nInput) * -4.0,
        objFunc=_gaussian_energy,
        optType="min",
        name="GaussianEnergy",
    )


def _shifted_quadratic(X):
    X = np.asarray(X)
    target = np.array([0.75, -0.5])
    return np.sum((X - target) ** 2, axis=1, keepdims=True)


def _anisotropic_quadratic(X):
    X = np.asarray(X)
    weights = np.array([0.25, 4.0])
    return np.sum(weights * X**2, axis=1, keepdims=True)


def _negative_quadratic_score(X):
    X = np.asarray(X)
    target = np.array([-0.4, 0.6])
    return -np.sum((X - target) ** 2, axis=1, keepdims=True)


def _assert_gaussian_result(res, nChains, nInput):
    assert res.decs.shape == (nChains, res.iters + 1, nInput)
    assert res.objs.shape == (nChains, res.iters + 1, 1)
    assert res.logProb.shape == (nChains, res.iters + 1)
    assert np.all(np.isfinite(res.decs))
    assert np.all(np.isfinite(res.logProb))
    assert np.allclose(res.logProb, -res.objs[..., 0])

    # These are intentionally loose. The test checks that the samplers move into
    # the high-probability basin of a known scalar target, not exact convergence.
    assert float(np.ravel(res.bestObjs)[0]) < 1.0
    assert np.linalg.norm(np.ravel(res.bestDecs)) < 1.5
    assert np.all((res.acceptanceRate >= 0.0) & (res.acceptanceRate <= 1.0))


def _assert_best_close(res, target, tol):
    assert np.linalg.norm(np.ravel(res.bestDecs) - np.asarray(target)) < tol


def test_inference_algorithms_on_gaussian_energy_benchmark():
    problem = _small_gaussian_problem(nInput=2)
    algorithms = [
        (MH(nChains=3, warmUp=5, maxIters=35, verboseFlag=False, saveFlag=False), 3),
        (AMH(nChains=3, warmUp=5, maxIterTimes=35, verboseFlag=False, saveFlag=False), 3),
        (MH_Gibbs(nChains=3, warmUp=5, maxIters=35, verboseFlag=False, saveFlag=False), 3),
        (DEMC(nChains=4, warmUp=5, maxIterTimes=35, verboseFlag=False, saveFlag=False), 4),
        (DREAM_ZS(nChains=4, warmUp=5, maxIters=35, archSize=3, verboseFlag=False, saveFlag=False), 4),
    ]

    for alg, nChains in algorithms:
        res = alg.run(problem, gamma=0.2, seed=2024)
        _assert_gaussian_result(res, nChains=nChains, nInput=2)


def test_inference_algorithms_on_shifted_quadratic_benchmark():
    problem = Problem(
        nInput=2,
        nObj=1,
        ub=[2.0, 2.0],
        lb=[-2.0, -2.0],
        objFunc=_shifted_quadratic,
        optType="min",
        name="ShiftedQuadratic",
    )
    algorithms = [
        (MH(nChains=3, warmUp=8, maxIters=45, verboseFlag=False, saveFlag=False), 0.8),
        (AMH(nChains=3, warmUp=8, maxIterTimes=45, verboseFlag=False, saveFlag=False), 0.8),
        (MH_Gibbs(nChains=3, warmUp=8, maxIters=45, verboseFlag=False, saveFlag=False), 0.9),
        (DEMC(nChains=4, warmUp=8, maxIterTimes=45, verboseFlag=False, saveFlag=False), 0.9),
        (DREAM_ZS(nChains=4, warmUp=8, maxIters=45, archSize=3, verboseFlag=False, saveFlag=False), 1.0),
    ]

    for alg, tol in algorithms:
        res = alg.run(problem, gamma=0.25, seed=2025)
        _assert_best_close(res, target=[0.75, -0.5], tol=tol)
        assert float(np.ravel(res.bestObjs)[0]) < 0.6


def test_inference_opt_type_max_score_semantics():
    problem = Problem(
        nInput=2,
        nObj=1,
        ub=[2.0, 2.0],
        lb=[-2.0, -2.0],
        objFunc=_negative_quadratic_score,
        optType="max",
        name="NegativeQuadraticScore",
    )

    res = MH(
        nChains=3,
        warmUp=8,
        maxIters=45,
        verboseFlag=False,
        saveFlag=False,
    ).run(problem, gamma=0.25, seed=99)

    _assert_best_close(res, target=[-0.4, 0.6], tol=0.8)
    # bestObjs are reported back in the user's original score direction.
    assert float(np.ravel(res.bestObjs)[0]) > -0.6
    # Trace objs store the internal oriented objective, so logProb is -objs.
    assert np.allclose(res.logProb, -res.objs[..., 0])


def test_inference_anisotropic_quadratic_benchmark():
    problem = Problem(
        nInput=2,
        nObj=1,
        ub=[3.0, 3.0],
        lb=[-3.0, -3.0],
        objFunc=_anisotropic_quadratic,
        optType="min",
        name="AnisotropicQuadratic",
    )

    res = AMH(
        nChains=3,
        warmUp=8,
        maxIterTimes=45,
        verboseFlag=False,
        saveFlag=False,
    ).run(problem, gamma=0.2, seed=77)

    assert float(np.ravel(res.bestObjs)[0]) < 0.5
    assert abs(float(np.ravel(res.bestDecs)[1])) < 0.6


def test_inference_hard_constraint_keeps_trace_feasible():
    def con_func(X):
        X = np.asarray(X)
        return (X[:, [0]] - 0.25)

    problem = Problem(
        nInput=2,
        nObj=1,
        nCon=1,
        ub=[1.0, 1.0],
        lb=[-1.0, -1.0],
        objFunc=_gaussian_energy,
        conFunc=con_func,
        optType="min",
        name="ConstrainedGaussianEnergy",
    )

    res = MH(
        nChains=3,
        warmUp=3,
        maxIters=20,
        verboseFlag=False,
        saveFlag=False,
        maxInitAttempts=50,
    ).run(problem, gamma=0.5, seed=9)

    assert res.cons.shape == (3, 20, 1)
    assert np.all(res.cons <= 1e-12)
    assert np.all(res.feasibleMask)
    assert res.bestFeasible is True


def test_inference_hard_constraint_with_narrow_feasible_region():
    def con_func(X):
        X = np.asarray(X)
        return np.hstack([
            X[:, [0]] - 0.1,
            -X[:, [0]] - 0.6,
        ])

    problem = Problem(
        nInput=2,
        nObj=1,
        nCon=2,
        ub=[1.0, 1.0],
        lb=[-1.0, -1.0],
        objFunc=_gaussian_energy,
        conFunc=con_func,
        optType="min",
        name="NarrowConstrainedGaussian",
    )

    res = MH_Gibbs(
        nChains=3,
        warmUp=3,
        maxIters=25,
        verboseFlag=False,
        saveFlag=False,
        maxInitAttempts=100,
    ).run(problem, gamma=0.4, seed=15)

    assert res.cons.shape == (3, 25, 2)
    assert np.all(res.cons <= 1e-12)
    assert np.all(res.decs[..., 0] <= 0.1 + 1e-12)
    assert np.all(res.decs[..., 0] >= -0.6 - 1e-12)


def test_inference_sqlite_reader_roundtrip():
    problem = _small_gaussian_problem(nInput=2)
    work_dir = Path("Result") / "_inference_sqlite_roundtrip"
    work_dir.mkdir(parents=True, exist_ok=True)
    before = set((work_dir / "Result").glob("*.sqlite3")) if (work_dir / "Result").exists() else set()
    problem.workDir = str(work_dir)

    alg = MH(
        nChains=2,
        warmUp=2,
        maxIters=8,
        verboseFlag=False,
        logFlag=False,
        saveFlag=True,
        saveFreq=2,
    )
    res = alg.run(problem, gamma=0.2, seed=123)

    after = set((work_dir / "Result").glob("*.sqlite3"))
    db_files = sorted(after - before)
    assert len(db_files) == 1

    with InfReader(db_files[0]) as reader:
        loaded = reader.load_result()
        snapshots = reader.list_snapshots()
        members = reader.load_last_snapshot_members()

    assert loaded.decs.shape == res.decs.shape
    assert np.allclose(loaded.logProb, res.logProb)
    assert len(snapshots) >= 1
    assert len(members) == 2
