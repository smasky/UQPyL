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
    lr = LinearRegression(lossType="BadType")
    with pytest.raises(ValueError):
        lr.fit(np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]]))


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

