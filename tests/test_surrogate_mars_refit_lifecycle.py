"""Surrogate mars refit lifecycle.

Migrated from test_review_c06_c12.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from UQPyL.surrogate.mars import MARS
from UQPyL.surrogate.scaler import StandardScaler


# Regression source: test_review_c06_c12.py::testMarsRefitsAcrossDimensionsAndRecoversAfterFailure
@pytest.mark.parametrize("scaled", [False, True])
def testMarsRefitsAcrossDimensionsAndRecoversAfterFailure(scaled):
    def makeModel():
        return MARS(scalers=(StandardScaler(), StandardScaler())) if scaled else MARS()

    model = makeModel()
    rng = np.random.default_rng(3)
    for nInput in [1, 2, 1]:
        x = rng.uniform(-2, 2, (50, nInput))
        y = 2 * x[:, :1] + x.sum(axis=1, keepdims=True)
        model.fit(x, y)
        expected = makeModel().fit(x, y)
        np.testing.assert_allclose(model.predict(x), expected.predict(x), atol=1e-12)
        np.testing.assert_allclose(model.predict_deriv(x), expected.predict_deriv(x), atol=1e-12)
        with pytest.raises(ValueError):
            model.fit(x, y[:-1])
        with pytest.raises(RuntimeError):
            model.predict(x)
        assert model.forward_trace() is None
        assert model.pruning_trace() is None
        model.fit(x, y)
        np.testing.assert_allclose(model.predict(x), expected.predict(x), atol=1e-12)


# Regression source: test_review_c06_c12.py::testMarsRefitWithoutPruningDoesNotKeepOldTrace
def testMarsRefitWithoutPruningDoesNotKeepOldTrace():
    x = np.linspace(-2, 2, 50)[:, None]
    model = MARS().fit(x, x * x)
    assert model.pruning_trace() is not None
    model.setting.set("enable_pruning", False)
    model.fit(x, x * x)
    assert model.pruning_trace() is None
