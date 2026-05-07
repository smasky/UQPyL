import numpy as np
import pytest

from UQPyL.problem.sop.single_simple_problem import Sphere
from UQPyL.surrogate.scaler import StandardScaler

SVR = pytest.importorskip(
    "UQPyL.surrogate.svr.support_vector_machine",
    reason="SVR extension module is not available in this environment.",
).SVR


def test_svr_invalid_params_raise():
    with pytest.raises(ValueError):
        SVR(symbol="bad")
    with pytest.raises(ValueError):
        SVR(kernel="bad")


def test_svr_parameter_activation_and_default_tune_parameters():
    m = SVR(symbol="epsilon-SVR", kernel="rbf")

    assert m.isParameterActive("C")
    assert m.isParameterActive("gamma")
    assert m.isParameterActive("epsilon")
    assert not m.isParameterActive("nu")
    assert not m.isParameterActive("coe0")
    assert not m.isParameterActive("degree")
    assert not m.isParameterActive("maxIter")
    assert not m.isParameterActive("eps")
    assert m.getDefaultTuneParameters() == ["C", "gamma", "epsilon"]

    m.setKernel("polynomial")
    m.setSymbol("nu-SVR")

    assert m.isParameterActive("gamma")
    assert m.isParameterActive("coe0")
    assert m.isParameterActive("degree")
    assert m.isParameterActive("nu")
    assert not m.isParameterActive("epsilon")


def test_svr_fit_predict_smoke():
    x = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    model = SVR(kernel="rbf", C=1.0, gamma=0.5, epsilon=0.1)
    model.fit(x, y)

    pred = model.predict(x[:5])
    assert pred.shape == (5, 1)
    assert np.isfinite(pred).all()

