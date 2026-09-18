"""Numerical and model-level regressions from the kernel review."""
import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, kv

from UQPyL.surrogate.gp.kernel import RBF as GPRBF, Matern, RationalQuadratic
from UQPyL.surrogate.gp.kernel.c_kernel_ import Constant
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Guass, Exp, Cubic
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.rbf.kernel import ThinPlateSpline


@pytest.mark.parametrize('kernelClass', [Guass, Exp, Cubic])
def test_kriging_signed_differences_and_training_reconstruction(kernelClass):
    kernel = kernelClass(theta=0.5)
    differences = np.array([[0.2], [-0.2], [0.0], [3.0]])
    np.testing.assert_allclose(kernel(differences), kernel(-differences))
    assert np.all((kernel(differences) >= 0) & (kernel(differences) <= 1))
    xTrain = np.array([[0.0], [0.2], [0.7], [1.0]])
    yTrain = np.sin(3 * xTrain)
    model = KRG(kernel=kernel)
    model.fitModel(*model.prepareTrainingData(xTrain, yTrain))
    np.testing.assert_allclose(model.predict(xTrain), yTrain, atol=1e-8)


def test_kriging_cubic_product_support_and_positive_semidefinite_matrix():
    kernel = Cubic(theta=np.array([0.5, 2.0]))
    distances = np.array([[0.4, -0.2], [2.0, 0.1], [0.0, 0.6], [0.0, 0.0]])
    np.testing.assert_allclose(kernel(distances), [0.896 * 0.648, 0, 0, 1])
    points = np.random.default_rng(123).random((20, 2))
    distances = np.column_stack([pdist(points[:, [i]]) for i in range(2)])
    matrix = squareform(Cubic(theta=1.0)(distances)) + np.eye(20)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12


def test_thin_plate_spline_zero_readonly_distances_and_tail():
    kernel = ThinPlateSpline(epsilon=2.0)
    distances = np.array([0.0, 0.5, 1.0])
    distances.flags.writeable = False
    np.testing.assert_allclose(kernel.evaluate(distances), [0, 0, 4 * np.log(2)])
    np.testing.assert_array_equal(distances, [0, 0.5, 1])
    points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    hasTail, tail = kernel.get_Tail_Matrix(points)
    assert hasTail
    np.testing.assert_array_equal(tail, np.column_stack([points, np.ones(4)]))
    assert kernel.get_A_Matrix(points).shape == (7, 7)


def test_thin_plate_spline_real_fit_reproduces_affine_function():
    points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.3, .7]])
    targets = (2 * points[:, 0] - 3 * points[:, 1] + 1)[:, None]
    model = RBF(kernel=ThinPlateSpline())
    model.fit(points, targets)
    query = np.array([[.2, .4], [.8, .3]])
    np.testing.assert_allclose(model.predict(query),
                               (2 * query[:, 0] - 3 * query[:, 1] + 1)[:, None], atol=1e-10)


def test_constant_self_and_rectangular_cross_covariance():
    kernel = Constant(c=2.5)
    np.testing.assert_array_equal(kernel(np.zeros((3, 2))), np.full((3, 3), 2.5))
    np.testing.assert_array_equal(kernel(np.zeros((3, 2)), np.ones((2, 2))),
                                  np.full((3, 2), 2.5))


@pytest.mark.parametrize('nu', [0.5, 1.5, 2.5, np.inf])
def test_matern_optimized_nu_encoding_clone_and_tuning(nu):
    kernel = Matern(nu=nu, optimize_nu=True)
    assert kernel.setting.get('nu') == nu
    cloned = kernel.clone()
    assert cloned.setting.get('nu') == nu
    points = np.array([[0.], [.3], [1.]])
    np.testing.assert_allclose(cloned(points), Matern(nu=nu)(points))
    infos, _, _ = cloned.setting.getParaInfos(['nu'])
    for encoded, actual in zip([.125, .375, .625, .875], [.5, 1.5, 2.5, np.inf]):
        cloned.setting.setVals(infos, np.array([encoded]))
        assert cloned.setting.get('nu') == actual
        np.testing.assert_allclose(cloned(points), Matern(nu=actual)(points))
    assert kernel.setting.get('nu') == nu


@pytest.mark.parametrize('nu', [0, -1, np.nan, -np.inf])
def test_matern_rejects_invalid_nu(nu):
    with pytest.raises(ValueError, match='nu'):
        Matern(nu=nu)


def test_matern_rejects_non_candidate_for_optimized_nu():
    with pytest.raises(ValueError, match='nu'):
        Matern(nu=0.7, optimize_nu=True)


@pytest.mark.parametrize('nu', [0.01, 0.1, 0.7, 3.7, 50., 100.])
def test_matern_general_nu_duplicate_points_and_reference(nu):
    points = np.array([[0.], [0.], [.3], [1.]])
    kernel = Matern(nu=nu)
    matrix = kernel(points, points)
    assert np.isfinite(matrix).all()
    np.testing.assert_array_equal(matrix[:2, :2], np.ones((2, 2)))
    np.testing.assert_allclose(kernel(points), matrix)
    distance = np.sqrt(2 * nu) * .3
    expected = 2**(1 - nu) / gamma(nu) * distance**nu * kv(nu, distance)
    np.testing.assert_allclose(matrix[0, 2], expected, rtol=1e-11)


@pytest.mark.parametrize('kernelClass,param,attr', [
    (GPRBF, 'l', 'length_attr'), (Matern, 'l', 'length_attr'),
    (RationalQuadratic, 'l', 'length_attr'),
    (Guass, 'theta', 'theta_attr'), (Exp, 'theta', 'theta_attr'),
    (Cubic, 'theta', 'theta_attr')])
@pytest.mark.parametrize('heterogeneous', [False, True])
def test_fixed_parameters_initialize_without_becoming_tunable(kernelClass, param, attr, heterogeneous):
    kernel = kernelClass(**{attr: None, 'heterogeneous': heterogeneous})
    expected = kernel.setting.get(param)
    kernel.initialize(3)
    assert param not in kernel.setting.getParaList()
    assert param in kernel.setting.parCon
    np.testing.assert_allclose(kernel.setting.get(param), np.full(3 if heterogeneous else 1, expected))
    kernel.initialize(3)
    cloned = kernel.clone()
    assert param not in cloned.setting.getParaList()
    np.testing.assert_array_equal(cloned.setting.get(param), kernel.setting.get(param))


@pytest.mark.parametrize('modelClass,kernelClass,param,attr', [
    (KRG, Guass, 'theta', 'theta_attr'), (KRG, Exp, 'theta', 'theta_attr'),
    (KRG, Cubic, 'theta', 'theta_attr')])
def test_all_fixed_kriging_parameters_support_public_fit(modelClass, kernelClass, param, attr):
    points = np.array([[0.], [.2], [.7], [1.]])
    targets = np.sin(3 * points)
    model = modelClass(kernel=kernelClass(**{attr: None}))
    model.fit(points, targets)
    assert model.getParaList() == []
    assert param in model.setting.parCon
    np.testing.assert_allclose(model.predict(points), targets, atol=1e-8)


def test_all_fixed_gp_parameters_support_public_fit():
    from UQPyL.surrogate.gp import GPR
    points = np.array([[0.], [.2], [.7], [1.]])
    targets = np.sin(3 * points)
    model = GPR(kernel=GPRBF(length_attr=None), C_attr=None, C=1e-12)
    model.fit(points, targets)
    assert model.getParaList() == []
    np.testing.assert_allclose(model.predict(points), targets, atol=1e-8)


# Independently generated with mpmath at 70 decimal digits using
# 2**(1-nu)/gamma(nu) * z**nu * besselk(nu,z), z=sqrt(2*nu)*distance.
# Keep constants here so tests do not add a runtime mpmath dependency.
@pytest.mark.parametrize('nu,distance,expected', [
    (.01, 1e-8, .3362513058656159763798702513242151742168),
    (.7, .3, .8081896193626388095519159984477839236585),
    (3.7, 1e-8, .999999999999999931481481481481485209695),
    (49.9, 1e-5, .9999999999489775051138032593892118492207),
    (50, 1e-5, .9999999999489795918380633503401124966831),
    (50, .3, .9551408787171346214324759064952030058986),
    (100, .01, .9999494962378669579637434436487591965988),
    (100, 1, .6042555686374475840060249408167592384401),
    (1000, .3, .9559553915814348268650348586511180565644),
    (1000, 10, 5.959301680326338715431965925698044793722e-22),
    (1e6, 1, .6065304322636281340190031513348638525922),
])
def test_matern_against_high_precision_reference(nu, distance, expected):
    actual = Matern(nu=nu)(np.array([[0.]]), np.array([[distance]]))[0, 0]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize('nu', [1e-12, .01, .7, 1., 1.00000001, 3.7, 49.9,
                               50., 100., 1000., 1e6, 1e20, 1e300])
def test_matern_wide_range_is_finite_bounded_and_decreasing(nu):
    import warnings
    distances = np.concatenate(([0.], np.logspace(-12, 3, 40)))[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = Matern(nu=nu)(np.zeros((1, 1)), distances).ravel()
    assert np.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    assert actual[0] == 1
    assert np.all(np.diff(actual) <= 1e-12)


@pytest.mark.parametrize('nu', [50., 100., 1000., 1e6])
def test_matern_large_order_covariance_and_gp_fit(nu):
    from UQPyL.surrogate.gp import GPR
    points = np.array([[0.], [1e-8], [.2], [.7], [1.]])
    kernel = Matern(nu=nu)
    matrix = kernel(points)
    np.testing.assert_allclose(matrix, kernel(points, points), atol=1e-13)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12
    model = GPR(kernel=kernel)
    model.fitModel(*model.prepareTrainingData(points, np.sin(points)))
    prediction, variance = model.predict(points, returnVar=True)
    assert np.isfinite(prediction).all()
    assert np.isfinite(variance).all()
    np.testing.assert_allclose(prediction, np.sin(points), atol=1e-6)


@pytest.mark.parametrize('nu', [1e20, 1e300])
def test_matern_large_finite_order_converges_to_gaussian_limit(nu):
    points = np.array([[0.], [1e-5], [.3], [1.], [10.]])
    np.testing.assert_allclose(Matern(nu=nu)(points), GPRBF()(points), rtol=2e-12, atol=1e-15)
