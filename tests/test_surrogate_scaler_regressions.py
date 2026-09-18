import numpy as np
import pytest

from UQPyL.surrogate.scaler import StandardScaler, MinMaxScaler
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.kriging import KRG

FACTORIES = [lambda: StandardScaler(), lambda: StandardScaler(3, 2),
             lambda: MinMaxScaler(), lambda: MinMaxScaler(-2, 3)]


@pytest.mark.parametrize('factory', FACTORIES)
def test_affine_inverse_mean_std_variance_and_no_mutation(factory):
    x = np.array([[2., 10.], [4., 30.], [8., 40.]])
    scaler = factory().fit(x)
    z = scaler.transform(x)
    np.testing.assert_allclose(scaler.inverse_transform(z), x)
    # Directly recover the affine slope, independently of stored scale fields.
    slope = scaler.inverse_transform(np.ones((1, 2))) - scaler.inverse_transform(np.zeros((1, 2)))
    variance = np.array([[.25, 4.], [1., 9.]])
    saved = variance.copy()
    np.testing.assert_allclose(scaler.inverse_transform_var(variance), variance*slope**2)
    np.testing.assert_allclose(scaler.inverse_transform_std(np.sqrt(variance)), np.sqrt(variance)*abs(slope))
    np.testing.assert_array_equal(variance, saved)


@pytest.mark.parametrize('factory', FACTORIES)
@pytest.mark.parametrize('x', [np.array([[5., 9.]]), np.array([[5., 1.], [5., 3.]])])
def test_constant_and_singleton_columns_stay_finite_and_reversible(factory, x):
    scaler = factory()
    with np.errstate(divide='raise', invalid='raise'):
        z = scaler.fit_transform(x)
        np.testing.assert_allclose(scaler.inverse_transform(z), x)
        # A constant observation does not justify zero predictive uncertainty.
        assert np.all(scaler.inverse_transform_var(np.ones((1, 2))) > 0)
        query = x + 2
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(query)), query)


@pytest.mark.parametrize('factory', FACTORIES)
def test_scaler_validation_and_refit(factory):
    scaler = factory()
    with pytest.raises(RuntimeError, match='fitted'):
        scaler.transform([[1.]])
    for invalid in [np.empty((0, 2)), np.ones((2, 2, 2)), [[np.nan]], [[np.inf]]]:
        with pytest.raises(ValueError):
            scaler.fit(invalid)
    scaler.fit([[1., 2.], [2., 4.]])
    with pytest.raises(ValueError, match='feature count'):
        scaler.transform([[1.]])
    with pytest.raises(ValueError, match='nonnegative'):
        scaler.inverse_transform_var([[-1., 1.]])
    scaler.fit([[10.], [20.]])
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform([[15.]])), [[15.]])


@pytest.mark.parametrize('factory', [lambda: StandardScaler(sitaX=0), lambda: StandardScaler(sitaX=-1),
    lambda: StandardScaler(muX=np.nan), lambda: StandardScaler(sitaX=np.inf),
    lambda: MinMaxScaler(1, 1), lambda: MinMaxScaler(2, 1), lambda: MinMaxScaler(0, np.inf)])
def test_invalid_configuration_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def fitFixed(modelClass, scaler, x, y):
    model = modelClass(scalers=(None, scaler))
    model.fitModel(*model.prepareTrainingData(x, y))
    return model


@pytest.mark.parametrize('modelClass', [GPR, KRG])
@pytest.mark.parametrize('factory', FACTORIES)
def test_prediction_units_under_shift_and_tenfold_output_scaling(modelClass, factory):
    x = np.array([[0.], [.2], [.55], [.85], [1.]])
    y = np.sin(4*x) + .2*x
    query = np.array([[.13], [.4], [.72]])
    first = fitFixed(modelClass, factory(), x, y)
    second = fitFixed(modelClass, factory(), x, 10*y+25)
    mean, var = first.predict(query, returnVar=True)
    mean2, var2 = second.predict(query, returnVar=True)
    np.testing.assert_allclose(mean2, 10*mean+25, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(var2, 100*var, rtol=1e-7, atol=1e-10)
    _, std = first.predict(query, returnStd=True)
    _, std2 = second.predict(query, returnStd=True)
    np.testing.assert_allclose(std2, 10*std, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(std**2, var)
    assert np.all(var > 0)


@pytest.mark.parametrize('modelClass', [GPR, KRG])
@pytest.mark.parametrize('factory', FACTORIES+[lambda: None])
def test_multioutput_matches_independent_fixed_models(modelClass, factory):
    x = np.array([[0.], [.2], [.55], [.85], [1.]])
    y = np.column_stack((np.sin(4*x[:, 0]), 20*np.cos(3*x[:, 0])+50))
    query = np.array([[.13], [.4], [.72]])
    model = fitFixed(modelClass, factory(), x, y)
    mean, var = model.predict(query, returnVar=True)
    assert mean.shape == var.shape == (3, 2)
    for i in range(2):
        separate = fitFixed(modelClass, factory(), x, y[:, [i]])
        expectedMean, expectedVar = separate.predict(query, returnVar=True)
        np.testing.assert_allclose(mean[:, [i]], expectedMean, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(var[:, [i]], expectedVar, rtol=1e-7, atol=1e-10)


def test_mean_only_custom_scaler_has_clear_uncertainty_error():
    class MeanOnly:
        def fit_transform(self, y): return y/2
        def inverse_transform(self, y): return y*2
    x = np.array([[0.], [.5], [1.]])
    model = fitFixed(GPR, MeanOnly(), x, np.sin(x))
    assert np.all(np.isfinite(model.predict([[.2]])))
    with pytest.raises(NotImplementedError, match='inverse_transform_var'):
        model.predict([[.2]], returnVar=True)
