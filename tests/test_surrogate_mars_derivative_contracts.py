"""Surrogate mars derivative contracts.

Migrated from test_review_b01_b03.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from UQPyL.surrogate.mars import MARS
from UQPyL.surrogate.scaler import StandardScaler, MinMaxScaler, Scaler
from UQPyL.surrogate.poly import PolyFeature
from UQPyL.surrogate import MultiSurrogate


# Regression source: test_review_b01_b03.py::testMarsDerivativeMatchesPredictionInOriginalUnits
@pytest.mark.parametrize("xScaler", [None, StandardScaler(muX=2, sitaX=3), MinMaxScaler(-2, 4)])
@pytest.mark.parametrize("yScaler", [None, StandardScaler(muX=-1, sitaX=2), MinMaxScaler(2, 5)])
def testMarsDerivativeMatchesPredictionInOriginalUnits(xScaler, yScaler):
    x = np.random.default_rng(2).uniform([10, -20], [20, 30], (100, 2))
    y = np.column_stack((3 * x[:, 0] - 2 * x[:, 1], -4 * x[:, 0] + 0.5 * x[:, 1]))
    model = MultiSurrogate(2, [MARS(scalers=(xScaler, yScaler)), MARS(scalers=(xScaler, yScaler))]).fit(x, y)
    query = x[:4]
    expected = np.broadcast_to([[3.0, -4.0], [-2.0, 0.5]], (4, 2, 2))
    derivative = model.predict_deriv(query)
    np.testing.assert_allclose(derivative, expected, atol=1e-10)
    for column in range(2):
        delta = np.zeros_like(query)
        delta[:, column] = 1e-4
        finite = (model.predict(query + delta) - model.predict(query - delta)) / 2e-4
        np.testing.assert_allclose(derivative[:, column], finite, atol=1e-8)
    np.testing.assert_allclose(model.predict_deriv(query, [1, 0]), derivative[:, [1, 0]])
    np.testing.assert_allclose(model.predict_deriv(query, model.models_list[0].xlabels_[1]), derivative[:, [1]])
    with pytest.raises(ValueError, match="out of range"):
        model.predict_deriv(query, -1)
    with pytest.raises(ValueError):
        model.fit(x, y[:-1])
    with pytest.raises(RuntimeError):
        model.predict_deriv(query)


class IdentityScaler(Scaler):
    def fit(self, values):
        return self

    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


# Regression source: test_review_b01_b03.py::testMarsUnsupportedDerivativePreprocessingIsExplicit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"polyFeature": PolyFeature()},
        {"scalers": (IdentityScaler(), None)},
        {"scalers": (None, IdentityScaler())},
    ],
)
def testMarsUnsupportedDerivativePreprocessingIsExplicit(kwargs):
    x = np.linspace(10, 20, 30)[:, None]
    model = MARS(**kwargs).fit(x, 3 * x + 2)
    with pytest.raises(NotImplementedError, match="affine"):
        model.predict_deriv(x)


# Regression source: test_review_b01_b03.py::testMarsNonlinearDerivativeUsesTransformedCoordinates
@pytest.mark.parametrize("smooth", [False, True])
def testMarsNonlinearDerivativeUsesTransformedCoordinates(smooth):
    x = np.random.default_rng(3).uniform([10.0, -5.0], [20.0, 5.0], (160, 2))
    y = (np.maximum(x[:, 0] - 15, 0) * x[:, 1] + 2 * x[:, 0])[:, None]
    model = MARS(max_degree=2, smooth=smooth, scalers=(StandardScaler(), MinMaxScaler(-2, 3))).fit(x, y)
    query = np.array([[12.3, -2.1], [17.2, 3.1], [18.1, -1.4]])
    derivative = model.predict_deriv(query)
    for column in range(2):
        delta = np.zeros_like(query)
        delta[:, column] = 1e-5
        finite = (model.predict(query + delta) - model.predict(query - delta)) / 2e-5
        np.testing.assert_allclose(derivative[:, column], finite, atol=1e-6)
    assert abs(derivative[0, 0, 0] - derivative[1, 0, 0]) > 0.1
