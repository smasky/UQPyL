"""Surrogate fit and validation.

Migrated from test_review_a01_a06.py; original regression provenance is retained below.
"""

from types import SimpleNamespace
import numpy as np
import pytest
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF
from UQPyL.surrogate.scaler import StandardScaler, MinMaxScaler
from UQPyL.surrogate.metric import mse, r_square, nse
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.regression.linear_regression import LinearRegression


def fixedModel(**kwargs):
    return GPR(kernel=RBF(length_scale=0.3, length_attr=None), C_attr=None, **kwargs)


# Regression source: test_review_a01_a06.py::testScalerOwnership
@pytest.mark.parametrize("scalerClass", [StandardScaler, MinMaxScaler])
@pytest.mark.parametrize("axis", [0, 1])
def testScalerOwnership(scalerClass, axis):
    shared = scalerClass()
    scalers = [None, None]
    scalers[axis] = shared
    x = np.linspace(0, 1, 8)[:, None]
    y = np.sin(3 * x)
    a, b = fixedModel(scalers=scalers), fixedModel(scalers=scalers)
    a.fit(x, y)
    before = a.predict(x, returnVar=True)
    b.fit(10 + 10 * x, 10 + 20 * y)
    after = a.predict(x, returnVar=True)
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)
    assert not shared.fitted
    assert a.xScaler is not b.xScaler if axis == 0 else a.yScaler is not b.yScaler


# Regression source: test_review_a01_a06.py::testFailedRefitInvalidatesAndCanRecover
@pytest.mark.parametrize("stage", ["prepare", "fit", "input"])
def testFailedRefitInvalidatesAndCanRecover(monkeypatch, stage):
    x = np.linspace(0, 1, 8)[:, None]
    y = np.sin(3 * x)
    model = fixedModel(scalers=(StandardScaler(), None)).fit(x, y)
    before = model.predict(x)

    def fail(*args):
        model.fitState["partial"] = 1
        raise ValueError("injected failure")

    with monkeypatch.context() as patch:
        if stage != "input":
            patch.setattr(model, "_prepare_training_components" if stage == "prepare" else "fitHyper", fail)
        with pytest.raises(ValueError):
            model.fit(10 + 10 * x, y[:-1] if stage == "input" else y)
    assert not model.fitState
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(x)
    model.fit(x, y)
    np.testing.assert_allclose(model.predict(x), before)


# Regression source: test_review_a01_a06.py::testMetricMixedSingleOutputShapes
@pytest.mark.parametrize("metric", [mse, r_square, nse])
def testMetricMixedSingleOutputShapes(metric):
    y = np.arange(1.0, 5.0)
    np.testing.assert_allclose(metric(y, y[:, None]), metric(y[:, None], y[:, None]))
    np.testing.assert_allclose(metric(y[:, None], y), metric(y[:, None], y[:, None]))


# Regression source: test_review_a01_a06.py::testMetricRejectsIncompatibleData
@pytest.mark.parametrize("metric", [mse, r_square, nse])
@pytest.mark.parametrize("other", [np.zeros((3, 2)), np.zeros((2, 1)), np.zeros((3, 1, 1)), np.full((3, 1), np.nan)])
def testMetricRejectsIncompatibleData(metric, other):
    with pytest.raises(ValueError):
        metric(np.arange(3.0), other)


class CandidateOptimizer:
    def run(self, problem, seed=None):
        points = np.array([[0.1], [0.2]])
        scores = problem.evaluate(points).objs
        assert not np.isnan(scores).any()
        best = int(np.argmax(scores[:, 0]))
        return SimpleNamespace(bestDecs=points[best], bestObjs=scores[best])


def tune(tuner, entry, x, y, **kwargs):
    options = {"paraGrid": {"C": [0.1, 0.2]}} if entry == "gridTune" else {"paraList": ["C"]}
    return getattr(tuner, entry)(x, y, seed=1, tuneMode="joint", **options, **kwargs)


# Regression source: test_review_a01_a06.py::testTunerRejectsUndefinedValidation
@pytest.mark.parametrize("entry", ["gridTune", "optTune"])
@pytest.mark.parametrize("badData", ["one_test_point", "constant"])
def testTunerRejectsUndefinedValidation(entry, badData):
    x = np.arange(8.0)[:, None]
    y = np.ones_like(x) if badData == "constant" else x * x
    tuner = AutoTuner(LinearRegression(lossType="Ridge"), CandidateOptimizer())
    with pytest.raises(ValueError, match="validation"):
        tune(tuner, entry, x, y, ratio=50 if badData == "constant" else 10)


# Regression source: test_review_a01_a06.py::testTunerAllCandidatesFailClearly
@pytest.mark.parametrize("entry", ["gridTune", "optTune"])
@pytest.mark.parametrize("failure", ["exception", "nan"])
def testTunerAllCandidatesFailClearly(monkeypatch, entry, failure):
    x = np.arange(10.0)[:, None]
    y = x * x
    model = LinearRegression(lossType="Ridge")
    if failure == "exception":

        def fail(*args):
            raise np.linalg.LinAlgError("bad candidate")

        monkeypatch.setattr(model, "fitModel", fail)
    else:
        monkeypatch.setattr(model, "predict", lambda x: np.full((len(x), 1), np.nan))
    with pytest.raises(RuntimeError, match="No candidate"):
        tune(AutoTuner(model, CandidateOptimizer()), entry, x, y, ratio=30)
    assert not model.fitState
