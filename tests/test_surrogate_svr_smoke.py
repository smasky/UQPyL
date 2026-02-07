import numpy as np
import pytest

from UQPyL.problem.sop.single_simple_problem import Sphere
from UQPyL.surrogate.svr.support_vector_machine import SVR
from UQPyL.util.scaler import StandardScaler


def test_svr_invalid_params_raise():
    with pytest.raises(ValueError):
        SVR(symbol="bad")
    with pytest.raises(ValueError):
        SVR(kernel="bad")


def test_svr_fit_predict_smoke_on_sphere():
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
    rng = np.random.default_rng(123)
    X = rng.uniform(problem.lb, problem.ub, size=(30, problem.nInput))
    Y = problem.objFunc(X)

    m = SVR(
        scalers=(StandardScaler(0, 1), StandardScaler(0, 1)),
        symbol="epsilon-SVR",
        kernel="rbf",
        C=1.0,
        gamma=1.0,
        epsilon=0.01,
        maxIter=5000,
        eps=0.001,
    )
    m.fit(X, Y)
    pred = m.predict(X[:5])
    assert pred.shape == (5, 1)
    assert np.isfinite(pred).all()

