"""Surrogate public fit contracts.

Migrated from test_review_b04_b06.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from UQPyL.surrogate.mars import MARS
from UQPyL.surrogate.scaler import StandardScaler
from UQPyL.surrogate.poly import PolyFeature
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF, Matern


# Regression source: test_review_b04_b06.py::testMarsPublicResultsRequireValidFit
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize(
    "entry",
    ["predict", "predict_deriv", "transform", "score", "score_samples", "summary", "summary_feature_importances"],
)
def testMarsPublicResultsRequireValidFit(failed, entry):
    x = np.linspace(1, 5, 40)[:, None]
    y = 3 * x + 2
    model = MARS()
    if failed:
        model.fit(x, y)
        with pytest.raises(ValueError):
            model.fit(x, y[:-1])
    args = () if entry.startswith("summary") else ((x, y) if entry.startswith("score") else (x,))
    with pytest.raises(RuntimeError):
        getattr(model, entry)(*args)
    assert model.forward_trace() is None
    assert model.pruning_trace() is None
    model.fit(x, y)
    getattr(model, entry)(*args)


# Regression source: test_review_b04_b06.py::testMarsScoresMatchOriginalUnitPredictions
@pytest.mark.parametrize(
    "kwargs", [{}, {"scalers": (StandardScaler(), StandardScaler())}, {"polyFeature": PolyFeature()}]
)
def testMarsScoresMatchOriginalUnitPredictions(kwargs):
    x = np.linspace(10, 20, 45)[:, None]
    y = np.hstack((3 * x + 2, x * x + 10))
    for index in range(2):
        values = y[:, index : index + 1]
        model = MARS(**kwargs).fit(x, values)
        target = values + 1
        prediction = model.predict(x)
        np.testing.assert_allclose(model.score_samples(x, target), 1 - ((target - prediction) / target) ** 2)
        weights = np.linspace(1, 3, len(x))[:, None]
        expected = 1 - np.sum(weights * (target - prediction) ** 2) / np.sum(
            weights * (target - np.average(target, weights=weights.ravel(), axis=0)) ** 2
        )
        assert model.score(x, target, sample_weight=weights) == pytest.approx(expected)
        prepared = model._transformX(x)
        np.testing.assert_allclose(model._inverseTransformY(model.transform(prepared) @ model.coef_.T), prediction)
        with pytest.raises(ValueError, match="shape"):
            model.score_samples(x, np.hstack((target, target)))


# Regression source: test_review_b04_b06.py::testMarsScoresPreserveExplicitMissingMask
def testMarsScoresPreserveExplicitMissingMask():
    x = np.linspace(1, 8, 60)[:, None]
    x[::4] = np.nan
    y = np.where(np.isnan(x), 10.0, 3 * x + 2)
    model = MARS()
    model.setting.set("allow_missing", True)
    model.fit(x, y)
    query = np.array([[2.0], [123.0], [6.0]])
    mask = np.array([[False], [True], [False]])
    nanQuery = query.copy()
    nanQuery[mask] = np.nan
    prediction = model.predict(query, missing=mask)
    np.testing.assert_allclose(prediction, model.predict(nanQuery))
    assert abs(prediction[1, 0] - model.predict(query)[1, 0]) > 1
    target = prediction + 1
    np.testing.assert_allclose(model.score_samples(query, target, missing=mask), 1 - 1 / target**2)
    expected = 1 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean(axis=0)) ** 2)
    assert model.score(query, target, missing=mask) == pytest.approx(expected)
    np.testing.assert_array_equal(query, [[2.0], [123.0], [6.0]])
    with pytest.raises(ValueError, match="shape"):
        model.predict(query, missing=np.zeros((3, 2), dtype=bool))
    with pytest.raises(ValueError, match="marked as missing"):
        model.predict(nanQuery, missing=np.zeros_like(mask))


# Regression source: test_review_b04_b06.py::testMarsMissingWithPreprocessingExplicitlyUnsupported
@pytest.mark.parametrize("entry", ["predict", "predict_deriv", "score", "score_samples"])
def testMarsMissingWithPreprocessingExplicitlyUnsupported(entry):
    x = np.linspace(1, 8, 40)[:, None]
    model = MARS(scalers=(StandardScaler(), None)).fit(x, 3 * x + 2)
    model.setting.set("allow_missing", True)
    mask = np.zeros_like(x, dtype=bool)
    mask[0] = True
    args = (x, 3 * x + 2) if entry.startswith("score") else (x,)
    with pytest.raises(NotImplementedError, match="preprocessing"):
        getattr(model, entry)(*args, missing=mask)


# Regression source: test_review_b04_b06.py::testUnknownGridParametersRejectedBeforeCandidateFit
@pytest.mark.parametrize(
    "grid",
    [
        {"misspelled_parameter": [0.1]},
        {"l": [0.0], "misspelled_parameter": [0.1]},
        {"kernel": [0.5, 1.5], "misspelled_parameter": [0.1]},
    ],
)
def testUnknownGridParametersRejectedBeforeCandidateFit(grid, monkeypatch):
    x = np.linspace(0, 1, 16)[:, None]
    model = GPR()
    model.setKernelChoices([RBF(), Matern(optimize_nu=True)])

    def forbidden(*args):
        pytest.fail("unknown names must be rejected before fitting candidates")

    monkeypatch.setattr(model, "fitModel", forbidden)
    with pytest.raises(ValueError, match="misspelled_parameter"):
        AutoTuner(model).gridTune(x, np.sin(3 * x), paraGrid=grid, tuneMode="joint", ratio=25)


# Regression source: test_review_b04_b06.py::testInactiveParameterRecognizedAcrossStructureChoices
def testInactiveParameterRecognizedAcrossStructureChoices(monkeypatch):
    x = np.linspace(0, 1, 16)[:, None]
    model = GPR()
    model.setKernelChoices([RBF(), Matern(optimize_nu=True)])
    monkeypatch.setattr(
        "UQPyL.surrogate.auto_tuner.r_square",
        lambda actual, predicted: 2.0 if model.kernel.displayName == "RBF" else 1.0,
    )
    (nu, kernel), score = AutoTuner(model).gridTune(
        x, np.sin(3 * x), paraGrid={"nu": [0.625], "kernel": [1.5, 0.5]}, ratio=25, tuneMode="joint", seed=1
    )
    assert nu is None
    assert kernel.displayName == "RBF"
    assert score == 2.0
