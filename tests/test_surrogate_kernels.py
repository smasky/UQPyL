import numpy as np
import pytest

from UQPyL.problem.mop.ZDT import ZDT1
from UQPyL.problem.sop.single_simple_problem import Sphere


def _sample_from_problem(problem, n, seed=123):
    rng = np.random.default_rng(seed)
    lb = np.asarray(problem.lb).reshape(1, -1)
    ub = np.asarray(problem.ub).reshape(1, -1)
    return rng.random((n, problem.nInput)) * (ub - lb) + lb


def test_rbf_kernels_evaluate_and_a_matrix_shapes_on_sphere_points():
    from UQPyL.surrogate.rbf.kernel.cubic_kernel import Cubic
    from UQPyL.surrogate.rbf.kernel.gaussian_kernel import Gaussian
    from UQPyL.surrogate.rbf.kernel.linear_kernel import Linear
    from UQPyL.surrogate.rbf.kernel.multiquadric_kernel import Multiquadric
    from UQPyL.surrogate.rbf.kernel.thin_plate_spline_kernel import ThinPlateSpline

    problem = Sphere(nInput=3, ub=1.0, lb=-1.0)
    X = _sample_from_problem(problem, 6)

    kernels = [Cubic(), Gaussian(), Linear(), Multiquadric(), ThinPlateSpline()]
    # distance matrix
    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

    for k in kernels:
        Phi = k.evaluate(dist.copy())
        assert Phi.shape == dist.shape

    # get_A_Matrix covers Tail matrix branches for some kernels
    A = Cubic().get_A_Matrix(X)
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] >= X.shape[0]


def test_kriging_kernels_call_on_dimensionwise_pdist():
    from scipy.spatial.distance import pdist

    from UQPyL.surrogate.kriging.kernel.guass_kernel import Guass
    from UQPyL.surrogate.kriging.kernel.cubic_kernel import Cubic
    from UQPyL.surrogate.kriging.kernel.exp_kernel import Exp

    problem = Sphere(nInput=4, ub=1.0, lb=-1.0)
    X = _sample_from_problem(problem, 6)

    # Build D like KRG._initialize: (nPairs, nFeature) with per-dimension pdist
    nSample, nFeature = X.shape
    D = np.zeros((int(nSample * (nSample - 1) / 2), nFeature))
    for k in range(nFeature):
        D[:, k] = pdist(X[:, [k]], metric="euclidean")

    kernels = [Guass(heterogeneous=True, theta=0.1), Cubic(heterogeneous=True, theta=0.1), Exp(heterogeneous=True, theta=0.1)]
    for ker in kernels:
        ker.initialize(nFeature)
        r = ker(D)
        assert r.shape == (D.shape[0],)
        assert np.all(np.isfinite(r))


def test_gp_kernels_matern_and_rq_basic_properties_on_zdt_points():
    from UQPyL.surrogate.gp.kernel.matern_kernel import Matern
    from UQPyL.surrogate.gp.kernel.rq_kernel import RationalQuadratic

    problem = ZDT1(nInput=6, ub=1.0, lb=0.0)
    X = _sample_from_problem(problem, 8)

    for nu in [0.5, 1.5, 2.5, np.inf]:
        k = Matern(nu=nu)
        K = k(X)
        assert K.shape == (X.shape[0], X.shape[0])
        assert np.allclose(np.diag(K), 1.0)

    rq = RationalQuadratic(length_scale=1.0, alpha=1.0)
    K2 = rq(X)
    assert K2.shape == (X.shape[0], X.shape[0])
    assert np.allclose(np.diag(K2), 1.0)


def test_gp_compiled_kernels_import_optional():
    # These are C-extension wrappers in some environments; skip if not available.
    pytest.importorskip("UQPyL.surrogate.gp.kernel.c_kernel_")
    pytest.importorskip("UQPyL.surrogate.gp.kernel.dot_kernel_")

