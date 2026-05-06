import numpy as np
import pytest

from UQPyL.problem.sop.single_simple_problem import Sphere
from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.util.scaler import StandardScaler


def test_linear_regression_ridge_fit_predict_on_sphere():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    rng = np.random.default_rng(123)
    X = rng.uniform(problem.lb, problem.ub, size=(30, problem.nInput))
    Y = problem.objFunc(X)

    lr = LinearRegression(
        scalers=(StandardScaler(0, 1), StandardScaler(0, 1)),
        lossType="Ridge",
        fitIntercept=True,
        C=1e-3,
    )
    lr.fit(X, Y)
    pred = lr.predict(X[:5])
    assert pred.shape == (5, 1)
    assert np.isfinite(pred).all()


def test_linear_regression_invalid_loss_type_raises():
    with pytest.raises(ValueError):
        LinearRegression(lossType="BadType")


def test_linear_regression_default_tune_parameters_and_activation():
    model = LinearRegression(lossType="Origin", fitIntercept=True)

    assert model.getDefaultTuneParameters() == []
    assert model.getDefaultTuneParameters(advanced=True) == ["lossType"]
    assert not model.isParameterActive("C")
    assert not model.isParameterActive("maxIter")

    model.setLossType("Ridge")
    assert model.isParameterActive("C")
    assert model.getDefaultTuneParameters(advanced=True) == ["lossType", "C"]

    model.setLossType("Lasso")
    assert model.isParameterActive("C")
    assert model.isParameterActive("maxIter")
    assert model.isParameterActive("maxEpoch")
    assert model.isParameterActive("tol")
    assert model.isParameterActive("p0")


def test_linear_regression_apply_parameter_values_updates_model_context():
    model = LinearRegression(lossType="Origin")

    model.applyParameterValues(["lossType", "C"], [1.2, np.log(1e-2)])

    assert model.lossType == "Ridge"
    assert model.setting.getVals("lossType") == "Ridge"
    assert np.isclose(model.setting.getVals("C"), 1e-2)
    assert model.isParameterActive("C")


def test_linear_regression_lasso_smoke_if_available():
    # Lasso uses a compiled extension in this repo; skip if not importable.
    pytest.importorskip("UQPyL.surrogate.regression.lasso")

    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    rng = np.random.default_rng(123)
    X = rng.uniform(problem.lb, problem.ub, size=(20, problem.nInput))
    Y = problem.objFunc(X)

    lr = LinearRegression(
        scalers=(StandardScaler(0, 1), StandardScaler(0, 1)),
        lossType="Lasso",
        fitIntercept=True,
        C=1e-2,
        maxIter=50,
        maxEpoch=2000,
        tolerance=1e-3,
        p0=5,
    )
    lr.fit(X, Y)
    pred = lr.predict(X[:3])
    assert pred.shape == (3, 1)


def test_linear_regression_ridge_matches_simple_linear_trend():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    Y = 1.0 + 2.0 * X

    lr = LinearRegression(lossType="Ridge", fitIntercept=True, C=1e-8)
    lr.fit(X, Y)

    pred = lr.predict(np.array([[4.0]]))
    assert pred.shape == (1, 1)
    assert np.isclose(pred[0, 0], 9.0, atol=1e-3)

