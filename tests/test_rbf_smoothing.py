"""Smoothing preserves polynomial constraints and matches an independent solver."""

import numpy as np
import pytest
from scipy.interpolate import RBFInterpolator

from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.rbf.kernel import Cubic, Gaussian, Linear, Multiquadric, ThinPlateSpline
from UQPyL.surrogate.scaler import StandardScaler


KERNELS = [(Cubic, "cubic", 1), (ThinPlateSpline, "thin_plate_spline", 1),
           (Linear, "linear", 0), (Multiquadric, "multiquadric", 0),
           (Gaussian, "gaussian", -1)]


def referenceModel(kernelClass, scipyName, degree, epsilon, smooth, X, Y):
    # UQPyL Gaussian uses exp(-epsilon*r**2); SciPy uses exp(-(epsilon*r)**2).
    scipyEpsilon = np.sqrt(epsilon) if kernelClass is Gaussian else epsilon
    return RBFInterpolator(X, Y, kernel=scipyName, degree=degree,
                           epsilon=scipyEpsilon, smoothing=smooth)


@pytest.mark.parametrize("kernelClass,scipyName,degree", KERNELS)
@pytest.mark.parametrize("smooth", [0.0, 0.1, 10.0])
@pytest.mark.parametrize("epsilon", [0.7, 1.8])
@pytest.mark.parametrize("nInputs", [1, 2])
def test_predictions_match_scipy_with_matching_kernel_conventions(
        kernelClass, scipyName, degree, smooth, epsilon, nInputs):
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 7).reshape(-1, 1) if nInputs == 1 else rng.uniform(-1, 1, (10, 2))
    Y = (np.sin(3 * X[:, 0]) + 0.2 * np.sum(X**2, axis=1)).reshape(-1, 1)
    probes = np.vstack([X, rng.uniform(-1.2, 1.2, (20, nInputs))])
    model = RBF(kernel=kernelClass(epsilon=epsilon), C_smooth=smooth).fit(X, Y)
    expected = referenceModel(kernelClass, scipyName, degree, epsilon, smooth, X, Y)(probes)
    np.testing.assert_allclose(model.predict(probes), expected, rtol=1e-7, atol=1e-8)
    if smooth == 0:
        np.testing.assert_allclose(model.predict(X), Y, rtol=1e-7, atol=1e-8)
    if degree >= 0:
        tail = np.column_stack([X, np.ones(len(X))]) if degree == 1 else np.ones((len(X), 1))
        np.testing.assert_allclose(tail.T @ model.fitState["coe_lambda"], 0, atol=1e-8)


@pytest.mark.parametrize("kernelClass", [Cubic, ThinPlateSpline, Linear, Multiquadric])
@pytest.mark.parametrize("smooth", [1.0, 100.0])
def test_smoothing_preserves_the_supported_polynomial_trend(kernelClass, smooth):
    X = np.linspace(0, 1, 12).reshape(-1, 1)
    probes = np.linspace(-0.2, 1.2, 31).reshape(-1, 1)
    affine = kernelClass in (Cubic, ThinPlateSpline)
    Y = 1 + 2 * X if affine else np.full_like(X, 3.0)
    expected = 1 + 2 * probes if affine else np.full_like(probes, 3.0)
    model = RBF(kernel=kernelClass(), C_smooth=smooth).fit(X, Y)
    np.testing.assert_allclose(model.predict(probes), expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(model.fitState["coe_lambda"], 0, atol=1e-10)


@pytest.mark.parametrize("kernelClass,scipyName,degree", KERNELS)
def test_large_smoothing_approaches_unpenalized_trend(kernelClass, scipyName, degree):
    X = np.linspace(-1, 1, 12).reshape(-1, 1)
    Y = np.sin(5 * X) + 2 + X
    model = RBF(kernel=kernelClass(), C_smooth=1e4).fit(X, Y)
    if degree == 1:
        tail = np.column_stack([X, np.ones(len(X))])
        expected = tail @ np.linalg.lstsq(tail, Y, rcond=None)[0]
    elif degree == 0:
        expected = np.full_like(Y, Y.mean())
    else:
        expected = np.zeros_like(Y)
    np.testing.assert_allclose(model.predict(X), expected, atol=0.005, rtol=0)


@pytest.mark.parametrize("kernelClass,scipyName,degree", KERNELS)
def test_positive_smoothing_handles_duplicate_points(kernelClass, scipyName, degree):
    X = np.array([[0.], [0.], [0.3], [0.7], [1.]])
    Y = np.array([[1.], [2.], [0.5], [-1.], [0.]])
    model = RBF(kernel=kernelClass(), C_smooth=0.1).fit(X, Y)
    expected = referenceModel(kernelClass, scipyName, degree, 1.0, 0.1, X, Y)(X)
    np.testing.assert_allclose(model.predict(X), expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("kernelClass,scipyName,degree", KERNELS)
@pytest.mark.parametrize("smooth", [0.0, 0.2])
def test_smoothing_supports_multiple_outputs_and_scaling(kernelClass, scipyName, degree, smooth):
    X = np.linspace(-2, 3, 12).reshape(-1, 1)
    Y = np.column_stack([10 + np.sin(X[:, 0]), 100 * np.cos(X[:, 0])])
    originalX, originalY = X.copy(), Y.copy()
    model = RBF(kernel=kernelClass(), C_smooth=smooth,
                scalers=(StandardScaler(), StandardScaler())).fit(X, Y)
    # StandardScaler in this package uses the sample standard deviation.
    scaledX = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    scaledY = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
    expected = referenceModel(kernelClass, scipyName, degree, 1.0, smooth, scaledX, scaledY)(scaledX)
    expected = expected * Y.std(axis=0, ddof=1) + Y.mean(axis=0)
    # The unregularized Gaussian system has condition number about 1.5e8;
    # allow roundoff from the existing LU/pseudoinverse solver at output scale 100.
    np.testing.assert_allclose(model.predict(X), expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(X, originalX)
    np.testing.assert_array_equal(Y, originalY)


@pytest.mark.parametrize("smooth", [-1.0, np.nan, np.inf, -np.inf, [0.1, 0.2], []])
def test_invalid_smoothing_is_rejected_before_solving(smooth):
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    with pytest.raises(ValueError, match="C_smooth"):
        RBF(C_smooth=smooth, C_smooth_attr=None).fit(X, X**2)


def test_smoothing_validation_also_covers_optimizer_updates():
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    model = RBF()
    model.applyParameterValues(["C_smooth"], [np.nan])
    with pytest.raises(ValueError, match="C_smooth"):
        model.fit(X, X**2)
