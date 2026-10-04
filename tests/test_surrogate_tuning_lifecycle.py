"""Surrogate tuning lifecycle.

Migrated from test_review_b01_b03.py; original regression provenance is retained below.
"""

from types import SimpleNamespace
import numpy as np
import pytest
from UQPyL.surrogate import MultiSurrogate
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.surrogate.scaler import StandardScaler, MinMaxScaler, Scaler


# Regression source: test_review_b01_b03.py::testTuningFailureInvalidatesAndAllowsRefit
@pytest.mark.parametrize("method", ["gridTune", "optTune"])
@pytest.mark.parametrize("stage", ["prepare", "parameter", "candidate", "refit", "interrupt"])
def testTuningFailureInvalidatesAndAllowsRefit(monkeypatch, method, stage):
    x = np.linspace(10, 20, 20)[:, None]
    y = 3 * x + 2
    model = LinearRegression(lossType="Ridge", scalers=(StandardScaler(), None))
    model.fit(x - 10, y)
    tuner = AutoTuner(model)

    def optimize(problem, seed):
        point = np.array([[np.log(0.1)]])
        return SimpleNamespace(bestDecs=point, bestObjs=problem.evaluate(point).objs)

    tuner.optimizer = SimpleNamespace(run=optimize)
    originalFit = model.fitModel
    originalPrepare = model.prepareTrainingData
    error = KeyboardInterrupt() if stage == "interrupt" else ValueError("injected failure")

    def fail(*args, **kwargs):
        raise error

    def prepare(a, b):
        result = originalPrepare(a, b)
        if stage == "prepare":
            raise error
        return result

    def fit(a, b):
        originalFit(a, b)
        if stage in ("candidate", "interrupt") or (stage == "refit" and len(a) == len(x)):
            raise error

    with monkeypatch.context() as patch:
        patch.setattr(model, "prepareTrainingData", prepare)
        patch.setattr(model, "fitModel", fit)
        if stage == "parameter":
            patch.setattr(model, "applyParameterValues", fail)
        args = dict(paraGrid={"C": [np.log(0.1)]}) if method == "gridTune" else dict(paraList=["C"])
        with pytest.raises(type(error)) as caught:
            getattr(tuner, method)(x, y, ratio=25, seed=1, tuneMode="joint", **args)
        assert caught.value is error
    assert model.fitState == {}
    assert model.xTrain is None and model.yTrain is None
    with pytest.raises(RuntimeError):
        model.predict(x)
    model.fit(x, y)
    assert np.all(np.isfinite(model.predict(x)))


# Regression source: test_review_b01_b03.py::testEmptyGridInvalidatesPreviousFit
def testEmptyGridInvalidatesPreviousFit():
    x = np.linspace(0, 1, 8)[:, None]
    model = LinearRegression(scalers=(StandardScaler(), None)).fit(x, 3 * x)
    with pytest.raises(ValueError, match="paraGrid"):
        AutoTuner(model).gridTune(x + 10, 3 * x, paraGrid={}, ratio=25, seed=1)
    with pytest.raises(RuntimeError):
        model.predict(x)


# Regression source: test_review_b01_b03.py::testOptimizerFailureInvalidatesPreviousFit
def testOptimizerFailureInvalidatesPreviousFit():
    x = np.arange(12.0)[:, None]
    model = LinearRegression(lossType="Ridge").fit(x, x)

    def run(**kwargs):
        raise RuntimeError("optimizer failure")

    with pytest.raises(RuntimeError, match="optimizer failure"):
        AutoTuner(model, SimpleNamespace(run=run)).optTune(x, x, paraList=["C"], ratio=25)
    with pytest.raises(RuntimeError):
        model.predict(x)


# Regression source: test_review_b01_b03.py::testDuplicateModelsRejectedWithoutChangingContainer
def testDuplicateModelsRejectedWithoutChangingContainer():
    model = LinearRegression()
    with pytest.raises(ValueError, match="distinct"):
        MultiSurrogate(2, [model, model])
    multi = MultiSurrogate(2)
    multi.append(model)
    with pytest.raises(ValueError, match="distinct"):
        multi.append(model)
    assert multi.models_list == [model]


# Regression source: test_review_b01_b03.py::testMutatedModelListRejectedBeforeUse
@pytest.mark.parametrize("method", ["fit", "predict"])
def testMutatedModelListRejectedBeforeUse(method, monkeypatch):
    first, second = LinearRegression(), LinearRegression()
    multi = MultiSurrogate(2, [first, second])
    x = np.arange(10.0)[:, None]
    y = np.hstack((3 * x + 2, -2 * x + 1))
    multi.fit(x, y)
    np.testing.assert_allclose(multi.predict(x), y, atol=1e-12)
    multi.models_list[1] = first

    def forbidden(*args):
        pytest.fail("duplicate validation must precede model use")

    monkeypatch.setattr(first, method, forbidden)
    with pytest.raises(ValueError, match="distinct"):
        getattr(multi, method)(x, y) if method == "fit" else multi.predict(x)


# Regression source: test_review_b01_b03.py::testSuccessfulTuningRefitsFullData
@pytest.mark.parametrize("method", ["gridTune", "optTune"])
def testSuccessfulTuningRefitsFullData(method):
    x = np.linspace(10, 20, 20)[:, None]
    y = 3 * x + 2
    model = LinearRegression(lossType="Ridge", scalers=(StandardScaler(), StandardScaler()))

    def optimize(problem, seed):
        point = np.array([[np.log(0.1)]])
        return SimpleNamespace(bestDecs=point, bestObjs=problem.evaluate(point).objs)

    tuner = AutoTuner(model, SimpleNamespace(run=optimize))
    args = dict(paraGrid={"C": [np.log(0.1)]}) if method == "gridTune" else dict(paraList=["C"])
    _, score = getattr(tuner, method)(x, y, ratio=25, seed=1, tuneMode="joint", **args)
    expected = LinearRegression(lossType="Ridge", C=0.1, scalers=(StandardScaler(), StandardScaler())).fit(x, y)
    assert np.isfinite(score)
    assert len(model.xTrain) == len(x)
    np.testing.assert_allclose(model.predict(x), expected.predict(x))
