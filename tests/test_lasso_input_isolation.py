from types import SimpleNamespace

import numpy as np
import pytest

from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.metric import r_square
from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.surrogate.regression.polynomial_regression import PolynomialRegression
from UQPyL.surrogate.scaler import StandardScaler

pytest.importorskip("UQPyL.surrogate.regression.lasso")


@pytest.mark.parametrize("layout", ["C", "F", "view", "readonly"])
@pytest.mark.parametrize("fitIntercept", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prepared_inputs_and_parent_arrays_survive_repeated_fits(layout, fitIntercept, dtype):
    X = np.random.default_rng(8).uniform(1, 3, (40, 3)).astype(dtype)
    Y = (X @ np.array([2., -0.8, 0.3], dtype=dtype) + 10).reshape(-1, 1)
    if layout == "F":
        X, Y = np.asfortranarray(X), np.asfortranarray(Y)
    parentX, parentY = X, Y
    originalX, originalY = X.copy(), Y.copy()
    if layout == "view":
        X, Y = X[::2], Y[::2]
    elif layout == "readonly":
        X.flags.writeable = Y.flags.writeable = False
    expectedX, expectedY = X.copy(), Y.copy()
    model = LinearRegression(lossType="Lasso", fitIntercept=fitIntercept, C=0.01,
                             maxEpoch=10000, tolerance=1e-5)
    predictions = []
    for _ in range(2):
        model.fitModel(X, Y)
        predictions.append(model.predict(expectedX))
        np.testing.assert_array_equal(X, expectedX)
        np.testing.assert_array_equal(Y, expectedY)
        np.testing.assert_array_equal(model.xTrain, expectedX)
        np.testing.assert_array_equal(model.yTrain, expectedY)
    np.testing.assert_array_equal(parentX, originalX)
    np.testing.assert_array_equal(parentY, originalY)
    np.testing.assert_array_equal(predictions[0], predictions[1])
    assert np.isfinite(predictions[0]).all()
    assert model.coef.dtype == dtype


@pytest.mark.parametrize("slope", [-2., 2.])
@pytest.mark.parametrize("penalty", [0.01, 2.])
def test_one_feature_solution_matches_closed_form(slope, penalty):
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    Y = 10 + slope * X
    centeredX, centeredY = X - X.mean(), Y - Y.mean()
    covariance = np.mean(centeredX * centeredY)
    coefficient = np.sign(covariance) * max(abs(covariance) - penalty, 0) / np.mean(centeredX**2)
    intercept = Y.mean() - X.mean() * coefficient
    model = LinearRegression(lossType="Lasso", C=penalty, tolerance=1e-8)
    probes = np.array([[-0.2], [0.5], [1.2]])
    for _ in range(2):
        model.fitModel(X, Y)
        np.testing.assert_allclose(model.coef, [coefficient], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(model.predict(probes), intercept + coefficient * probes,
                                   atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("modelClass", [LinearRegression, PolynomialRegression])
@pytest.mark.parametrize("entry", ["gridTune", "optTune"])
@pytest.mark.parametrize("scaled", [False, True])
def test_duplicate_tuning_candidates_have_identical_predictions_and_scores(modelClass, entry, scaled, capsys):
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    Y = 10 + 2 * X + 0.3 * X**2
    originalX, originalY = X.copy(), Y.copy()

    def makeModel():
        scalers = (StandardScaler(), StandardScaler()) if scaled else (None, None)
        return modelClass(lossType="Lasso", C=0.01, scalers=scalers, tolerance=1e-8)

    model = makeModel()
    predictions = []
    originalPredict = model.predict

    def predict(X):
        result = originalPredict(X)
        predictions.append(result.copy())
        return result

    model.predict = predict

    class RepeatingOptimizer:
        def run(self, problem, seed):
            candidates = np.full((2, 1), np.log(0.01))
            scores = problem.evaluate(candidates).objs
            np.testing.assert_array_equal(scores[0], scores[1])
            return SimpleNamespace(bestDecs=candidates[0], bestObjs=scores[0])

    tuner = AutoTuner(model, RepeatingOptimizer())
    kwargs = {"paraGrid": {"C": [np.log(0.01), np.log(0.01)]}} if entry == "gridTune" else {"paraList": ["C"]}
    best, score = getattr(tuner, entry)(X, Y, ratio=25, tuneMode="joint", seed=1, **kwargs)
    assert len(predictions) == 2
    np.testing.assert_array_equal(predictions[0], predictions[1])
    train, test = (tuner.lastSplit[key] for key in ["train_indices", "test_indices"])
    reference = makeModel().fit(X[train], Y[train])
    np.testing.assert_allclose(predictions[0], reference.predict(X[test]), rtol=1e-9, atol=1e-9)
    assert np.asarray(score).item() == pytest.approx(r_square(Y[test], predictions[0]))
    assert best == pytest.approx(0.01)
    reference.fit(X, Y)
    np.testing.assert_allclose(originalPredict(X), reference.predict(X), rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(X, originalX)
    np.testing.assert_array_equal(Y, originalY)
    assert "Warning:" not in capsys.readouterr().out


@pytest.mark.parametrize("modelClass", [LinearRegression, PolynomialRegression])
def test_grid_candidate_order_does_not_change_selected_model(modelClass):
    X = np.linspace(0, 1, 24).reshape(-1, 1)
    Y = 10 + 2 * X + 0.3 * X**2
    results = []
    for penalties in [[0.01, 0.1, 0.3], [0.3, 0.1, 0.01]]:
        model = modelClass(lossType="Lasso", tolerance=1e-8)
        best, score = AutoTuner(model).gridTune(X, Y, paraGrid={"C": np.log(penalties)},
                                               ratio=25, tuneMode="joint", seed=3)
        results.append((best, score, model.predict(X)))
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]
    np.testing.assert_array_equal(results[0][2], results[1][2])


def test_solver_failure_does_not_leave_inputs_centered(monkeypatch):
    X = np.asfortranarray(np.arange(12, dtype=float).reshape(6, 2))
    Y = 10 + X[:, :1]
    originalX, originalY = X.copy(), Y.copy()

    def fail(*args, **kwargs):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr("UQPyL.surrogate.regression.lasso.celer", fail)
    with pytest.raises(RuntimeError, match="injected solver failure"):
        LinearRegression(lossType="Lasso").fitModel(X, Y)
    np.testing.assert_array_equal(X, originalX)
    np.testing.assert_array_equal(Y, originalY)
