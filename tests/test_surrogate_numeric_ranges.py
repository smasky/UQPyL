"""Scores, preprocessing and uncertainty retain their meaning across units."""

import numpy as np
import pytest

from UQPyL.surrogate import AutoTuner, MinMaxScaler, MultiSurrogate, StandardScaler
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as GpRbf
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Exp
from UQPyL.surrogate.metric import r_square, nse
from UQPyL.surrogate.regression import PolynomialRegression


@pytest.mark.parametrize("metric", [r_square, nse])
@pytest.mark.parametrize("factor", [1.0, 1e-200, -1e-200, 1e160, -1e160])
def test_dimensionless_scores_match_hand_calculation(metric, factor):
    y = np.arange(1.0, 5.0)[:, None]
    prediction = y + np.array([-0.2, 0.2, -0.2, 0.2])[:, None]
    assert metric(y * factor, prediction * factor) == pytest.approx(0.968, abs=1e-14)


def test_score_preserves_output_weights_and_ignores_constant_column_scale():
    y = np.arange(1.0, 5.0)[:, None]
    error = np.array([-0.2, 0.2, -0.2, 0.2])[:, None]
    assert r_square(
        np.hstack([y * 1e-200, np.full_like(y, 1e200)]), np.hstack([(y + error) * 1e-200, np.full_like(y, 1e200)])
    ) == pytest.approx(0.968)
    assert r_square(np.hstack([y, 2 * y]), np.hstack([y + error, 2 * y])) == pytest.approx(1 - 0.16 / 25)
    with pytest.warns(RuntimeWarning):
        assert np.isnan(r_square(np.full((3, 1), 0.1), np.full((3, 1), 0.1)))


@pytest.mark.parametrize("factor", [1e-200, 1e160])
@pytest.mark.parametrize("mode", ["joint", "separate"])
def test_tuner_accepts_nonconstant_extreme_targets(factor, mode):
    x = np.linspace(-1, 1, 16)[:, None]
    tuner = AutoTuner(PolynomialRegression(scalers=(None, MinMaxScaler())))
    degree, score = tuner.gridTune(
        x,
        (x * x + 2 * x + 3) * factor,
        paraGrid={"degree": [2]},
        tuneMode=mode,
        splitIndices=(np.arange(12), np.arange(12, 16)),
    )
    assert degree == 2 and score == pytest.approx(1.0)
    np.testing.assert_allclose(tuner.model.predict(x) / factor, x * x + 2 * x + 3, atol=1e-12)


@pytest.mark.parametrize("factor", [1e-200, 1e160])
@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (3.0, 2.0)])
def test_standard_scaler_extreme_columns_match_sample_moments(factor, mu, sigma):
    base = np.arange(1.0, 5.0)[:, None]
    values = np.hstack([base * factor, np.full_like(base, 0.1 * factor)])
    scaler = StandardScaler(mu, sigma).fit(values)
    transformed = scaler.transform(values)
    np.testing.assert_allclose(transformed[:, 0], mu + sigma * (base[:, 0] - 2.5) / np.sqrt(5 / 3), atol=1e-14)
    np.testing.assert_allclose(transformed[:, 1], mu, atol=1e-14)
    np.testing.assert_allclose(scaler.inverse_transform(transformed) / factor, values / factor, atol=1e-14)


@pytest.mark.parametrize("value", [0.1, 1e-200, 1e308])
def test_standard_scaler_true_constant_stays_exact(value):
    scaler = StandardScaler().fit(np.full((7, 1), value))
    assert scaler.sita.item() == 0
    np.testing.assert_array_equal(scaler.transform(np.full((7, 1), value)), np.zeros((7, 1)))


def test_standard_scaler_opposite_extremes_roundtrip_without_intermediate_overflow():
    values = np.full((100, 1), -1e308)
    values[-1] = 1e308
    scaler = StandardScaler().fit(values)
    transformed = scaler.transform(values)
    assert np.all(np.isfinite(transformed))
    np.testing.assert_allclose(scaler.inverse_transform(transformed) / 1e308, values / 1e308, atol=1e-14)


@pytest.mark.parametrize("value", [1e-200, 1e160])
def test_minmax_nonzero_target_origin_preserves_constant_column(value):
    scaler = MinMaxScaler(-2, 3).fit(np.full((5, 1), value))
    transformed = scaler.transform(np.full((5, 1), value))
    np.testing.assert_array_equal(transformed, np.full((5, 1), -2.0))
    np.testing.assert_array_equal(scaler.inverse_transform(transformed), np.full((5, 1), value))


def fixedModel(family, scaler):
    if family == "GPR":
        return GPR(kernel=GpRbf(length_scale=0.3, length_attr=None), C=0.01, C_attr=None, scalers=(None, scaler))
    return KRG(kernel=Exp(theta=2.0, theta_attr=None), scalers=(None, scaler))


@pytest.mark.parametrize("family", ["GPR", "KRG"])
@pytest.mark.parametrize("factor", [1e-200, 1e160])
@pytest.mark.parametrize("scalerClass", [MinMaxScaler, StandardScaler])
@pytest.mark.parametrize("multi", [False, True])
def test_finite_standard_deviation_is_restored_without_squaring_scale(family, factor, scalerClass, multi):
    x = np.linspace(0, 1, 9)[:, None]
    y = 2 + np.sin(3 * x)
    query = np.array([[0.17], [1.4]])
    expectedMean, expectedStd = fixedModel(family, scalerClass()).fit(x, y).predict(query, returnStd=True)
    if multi:
        model = MultiSurrogate(2, [fixedModel(family, scalerClass()), fixedModel(family, scalerClass())])
        model.fit(x, np.hstack([y, y * factor]))
        mean, std = model.predict(query, returnStd=True)
        divisor = np.array([1.0, factor])
        np.testing.assert_allclose(mean / divisor, np.hstack([expectedMean, expectedMean]), rtol=1e-10)
        np.testing.assert_allclose(std / divisor, np.hstack([expectedStd, expectedStd]), rtol=1e-10)
    else:
        model = fixedModel(family, scalerClass()).fit(x, y * factor)
        mean, std = model.predict(query, returnStd=True)
        np.testing.assert_allclose(mean / factor, expectedMean, rtol=1e-10)
        np.testing.assert_allclose(std / factor, expectedStd, rtol=1e-10)
    with pytest.warns(RuntimeWarning, match="variance.*range"):
        _, variance = model.predict(query, returnVar=True)
    affected = variance[:, -1]
    assert np.all(affected == 0) if factor < 1 else np.all(np.isposinf(affected))


@pytest.mark.parametrize("scale,variance,expected", [(1e200, 1e-200, 1e200), (1e-200, 1e200, 1e-200)])
def test_representable_variance_survives_unrepresentable_scale_square(scale, variance, expected):
    scaler = MinMaxScaler().fit(np.array([[0.0], [scale]]))
    restored = scaler.inverse_transform_var(np.array([[variance], [0.0]]))
    assert restored[0, 0] / expected == pytest.approx(1.0)
    assert restored[1, 0] == 0


@pytest.mark.parametrize("entry", ["fit", "fitModel", "fitHyper"])
@pytest.mark.parametrize("bad", ["nan_y", "inf_y", "nan_x", "negative_c", "nan_c", "inf_c"])
def test_gpr_rejects_bad_data_before_kernel_and_invalidates_fit(entry, bad, monkeypatch):
    x = np.linspace(0, 1, 7)[:, None]
    y = np.sin(3 * x)
    model = fixedModel("GPR", None).fit(x, y)
    if bad.endswith("_y"):
        y[3] = np.nan if bad.startswith("nan") else np.inf
    elif bad == "nan_x":
        x[3] = np.nan
    else:
        model.setting.parCon["C"] = {"negative_c": -0.01, "nan_c": np.nan, "inf_c": np.inf}[bad]

    def unexpected(*args, **kwargs):
        pytest.fail("invalid inputs reached covariance calculation")

    monkeypatch.setattr(model, "_objfunc", unexpected)
    with pytest.raises(ValueError):
        getattr(model, entry)(x, y)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict([[0.5]])


def test_gpr_zero_noise_remains_valid_and_negative_search_bounds_rejected():
    x = np.array([[0.0], [0.3], [0.8], [1.0]])
    model = GPR(kernel=GpRbf(length_scale=0.2, length_attr=None), C=0, C_attr=None).fit(x, x * x)
    np.testing.assert_allclose(model.predict(x), x * x, atol=1e-13)
    bad = GPR(C_attr={"lb": -1.0, "ub": 1.0, "type": "float", "log": False})
    with pytest.raises(ValueError, match="bounds"):
        bad.fit(x, x * x)
