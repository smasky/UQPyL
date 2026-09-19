from types import SimpleNamespace

import numpy as np
import pytest

from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.scaler import StandardScaler, MinMaxScaler
from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.surrogate.poly import PolyFeature
from UQPyL.surrogate.metric import r_square


@pytest.mark.parametrize('entry', ['gridTune', 'optTune'])
@pytest.mark.parametrize('mode', ['joint', 'separate'])
@pytest.mark.parametrize('scalerClass', [StandardScaler, MinMaxScaler])
def test_tuner_fits_scalers_only_on_training_then_refits_full_data(monkeypatch, entry, mode, scalerClass):
    x = np.array([[0.], [1.], [2.], [3.], [100.], [200.]])
    y = 1+2*x+.3*x*x
    originalX, originalY = x.copy(), y.copy()
    train, test = np.arange(4), np.arange(4, 6)
    monkeypatch.setattr('UQPyL.surrogate.auto_tuner.RandSelect.split', lambda self, X, **kwargs: (train, test))
    class TrackingScaler(scalerClass):
        def __init__(self):
            super().__init__()
            self.fittedData = []
        def fit(self, X):
            self.fittedData.append(np.asarray(X).copy())
            return super().fit(X)
    sx, sy = TrackingScaler(), TrackingScaler()
    model = LinearRegression(scalers=(sx, sy), polyFeature=PolyFeature(degree=2), lossType='Ridge')
    # Models own independent scaler copies; inspect the installed components.
    sx, sy = model.xScaler, model.yScaler
    queries = []
    originalPredict = model.predict
    def predict(X):
        queries.append(np.asarray(X).copy())
        assert len(sx.fittedData) == len(sy.fittedData) == 1
        np.testing.assert_array_equal(sx.fittedData[0], x[train])
        np.testing.assert_array_equal(sy.fittedData[0], y[train])
        return originalPredict(X)
    model.predict = predict
    values = [1e-4, .1]
    class EnumeratingOptimizer:
        def run(self, problem, seed):
            candidates = np.asarray(values)[:, None]
            scores = problem.evaluate(candidates).objs
            best = np.argmax(scores[:, 0])
            return SimpleNamespace(bestDecs=candidates[best:best+1], bestObjs=scores[best:best+1])
    tuner = AutoTuner(model, EnumeratingOptimizer())
    kwargs = {'paraGrid': {'C': values}} if entry == 'gridTune' else {'paraList': ['C']}
    best, score = getattr(tuner, entry)(x, y, tuneMode=mode, **kwargs)
    assert len(queries) == 2
    for query in queries:
        np.testing.assert_array_equal(query, x[test])
    assert len(sx.fittedData) == len(sy.fittedData) == 2
    np.testing.assert_array_equal(sx.fittedData[1], x)
    np.testing.assert_array_equal(sy.fittedData[1], y)
    expectedScores = []
    expectedParameters = []
    for value in values:
        reference = LinearRegression(scalers=(scalerClass(), scalerClass()),
                                     polyFeature=PolyFeature(degree=2), lossType='Ridge')
        reference.applyParameterValues(['C'], [value])
        prepared = reference.prepareTrainingData(x[train], y[train])
        getattr(reference, 'fitModel' if mode == 'joint' else 'fitHyper')(*prepared)
        expectedScores.append(r_square(y[test], reference.predict(x[test])))
        expectedParameters.append(reference.getParameterValues("C"))
    assert float(np.asarray(score).ravel()[0]) == pytest.approx(max(expectedScores))
    np.testing.assert_allclose(best, expectedParameters[np.argmax(expectedScores)])
    final = LinearRegression(scalers=(scalerClass(), scalerClass()),
                             polyFeature=PolyFeature(degree=2), lossType='Ridge')
    final.applyParameterValues(['C'], [values[np.argmax(expectedScores)]])
    getattr(final, 'fitModel' if mode == 'joint' else 'fitHyper')(*final.prepareTrainingData(x, y))
    np.testing.assert_allclose(originalPredict(x), final.predict(x))
    np.testing.assert_array_equal(x, originalX)
    np.testing.assert_array_equal(y, originalY)
