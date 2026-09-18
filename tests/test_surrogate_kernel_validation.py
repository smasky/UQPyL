"""Kernel domain and dimensional validation at configuration/use boundaries."""
import numpy as np
import pytest

from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as GPRBF, Matern, RationalQuadratic
from UQPyL.surrogate.gp.kernel.c_kernel_ import Constant
from UQPyL.surrogate.gp.kernel.dot_kernel_ import DotProduct
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Guass, Exp, Cubic as KRGCubic
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.rbf.kernel import Cubic, Gaussian, Linear, Multiquadric, ThinPlateSpline


POSITIVE_CASES = [(GPRBF, 'length_scale'), (Matern, 'length_scale'),
                  (RationalQuadratic, 'length_scale'), (RationalQuadratic, 'alpha')]
POSITIVE_CASES += [(kernel, 'epsilon') for kernel in
                  [Cubic, Gaussian, Linear, Multiquadric, ThinPlateSpline]]


@pytest.mark.parametrize('kernelClass,parameter', POSITIVE_CASES)
@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf, -np.inf])
def test_positive_parameter_domains(kernelClass, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        kernelClass(**{parameter: value})


@pytest.mark.parametrize('kernelClass,parameter', [(Guass, 'theta'), (Exp, 'theta'),
    (KRGCubic, 'theta'), (Constant, 'c'), (DotProduct, 'sigma')])
@pytest.mark.parametrize('value', [-1., np.nan, np.inf])
def test_nonnegative_parameter_domains(kernelClass, parameter, value):
    with pytest.raises(ValueError, match='theta|constant|sigma'):
        kernelClass(**{parameter: value})


def test_valid_zero_parameters_and_numeric_scalars_are_preserved():
    points = np.array([[0., 1.], [1., 0.]])
    np.testing.assert_array_equal(Constant(c=0)(points), np.zeros((2, 2)))
    np.testing.assert_array_equal(DotProduct(sigma=0)(points), points @ points.T)
    for kernelClass in [Guass, Exp, KRGCubic]:
        np.testing.assert_array_equal(kernelClass(theta=0)(points), np.ones(2))
    for value in [1, np.float32(1), np.array(1.)]:
        np.testing.assert_allclose(GPRBF(length_scale=value)(points), GPRBF()(points))


@pytest.mark.parametrize('value', [[], [[1., 2.]], [1., np.nan], '1', True, 1+2j])
def test_length_parameter_rejects_invalid_structure_and_types(value):
    with pytest.raises(ValueError, match='length_scale'):
        GPRBF(length_scale=value)


@pytest.mark.parametrize('kernelClass,parameter', [(RationalQuadratic, 'alpha'),
    (Matern, 'nu'), (Gaussian, 'epsilon'), (Constant, 'c'), (DotProduct, 'sigma')])
def test_scalar_parameters_reject_vectors(kernelClass, parameter):
    with pytest.raises(ValueError, match='scalar'):
        kernelClass(**{parameter: [1., 2.]})


@pytest.mark.parametrize('kernelClass,parameter,attr', [(GPRBF, 'length_scale', 'length_attr'),
    (Matern, 'length_scale', 'length_attr'), (RationalQuadratic, 'length_scale', 'length_attr'),
    (Guass, 'theta', 'theta_attr'), (Exp, 'theta', 'theta_attr'), (KRGCubic, 'theta', 'theta_attr')])
@pytest.mark.parametrize('fixed', [True, False])
def test_feature_dimensions_checked_before_initialization_and_evaluation(kernelClass, parameter, attr, fixed):
    kwargs = {parameter: [1., 2.]}
    if fixed:
        kwargs[attr] = None
    kernel = kernelClass(**kwargs)
    with pytest.raises(ValueError, match='3.*features'):
        kernel.initialize(3)
    with pytest.raises(ValueError, match='3.*features'):
        kernel(np.zeros((4, 3)))
    kernel.initialize(2)
    assert np.isfinite(kernel(np.zeros((4, 2)))).all()


@pytest.mark.parametrize('kernel', [GPRBF(), Matern(), RationalQuadratic(), Constant(), DotProduct()])
def test_gp_input_shape_and_cross_feature_mismatch(kernel):
    with pytest.raises(ValueError, match='two-dimensional'):
        kernel(np.zeros(3))
    with pytest.raises(ValueError, match='same number of features'):
        kernel(np.zeros((3, 2)), np.zeros((4, 3)))


@pytest.mark.parametrize('kernelClass,parameter', [(GPRBF, 'l'), (Matern, 'nu'),
    (RationalQuadratic, 'alpha'), (Guass, 'theta'), (Exp, 'theta'), (KRGCubic, 'theta'),
    (Constant, 'constant'), (DotProduct, 'sigma'), (Gaussian, 'epsilon')])
def test_direct_parameter_mutation_is_checked_before_calculation(kernelClass, parameter):
    kernel = kernelClass()
    if parameter in kernel.setting.parCon:
        kernel.setting.parCon[parameter] = np.nan
    else:
        kernel.setting.parVal[parameter][...] = np.nan
    with pytest.raises(ValueError, match=parameter):
        if isinstance(kernel, Gaussian):
            kernel.evaluate(np.zeros((2, 2)))
        else:
            kernel(np.zeros((2, 1)))


@pytest.mark.parametrize('model,kernelParameter', [
    (GPR(kernel=GPRBF()), 'l'), (KRG(kernel=Exp()), 'theta'),
    (RBF(kernel=Gaussian()), 'epsilon')])
def test_optimizer_update_is_checked_before_model_fit(model, kernelParameter):
    # applyParameterValues uses encoded/log coordinates. Simulate an invalid
    # optimizer result and exercise the actual public fitting boundary.
    model.applyParameterValues([kernelParameter], [np.nan])
    with pytest.raises(ValueError, match=kernelParameter):
        model.fit(np.array([[0.], [.2], [.7], [1.]]), np.array([[0.], [.2], [.7], [1.]]))


@pytest.mark.parametrize('attr', [
    {'lb': 0., 'ub': 1.}, {'lb': .1, 'ub': np.inf},
    {'lb': 2., 'ub': 1.}, {'lb': np.nan, 'ub': 1.}])
def test_invalid_length_optimization_bounds(attr):
    with pytest.raises(ValueError, match='length_scale|l lower bound'):
        GPRBF(length_attr=attr)


def test_logarithmic_bounds_and_anisotropic_bound_dimensions():
    with pytest.raises(ValueError, match='logarithmic bounds'):
        Exp(theta_attr={'lb': 0., 'ub': 1., 'log': True})
    kernel = GPRBF(length_attr={'lb': [.1, .2], 'ub': [1., 2.]})
    with pytest.raises(ValueError, match='3.*features'):
        kernel.initialize(3)


@pytest.mark.parametrize('value', [np.nan, np.inf, -1., 2.])
def test_optimized_nu_rejects_invalid_choice_coordinates(value):
    kernel = Matern(optimize_nu=True)
    kernel.setting.parVal['nu'][...] = value
    with pytest.raises(ValueError, match='nu choice'):
        kernel(np.zeros((2, 1)))
