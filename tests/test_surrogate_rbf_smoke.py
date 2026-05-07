import numpy as np
import pytest

from UQPyL.surrogate.rbf.radial_basis_function import RBF
from UQPyL.surrogate.rbf.kernel import Cubic, Gaussian


def test_rbf_predict_protocol_smoke():
    x = np.array([[0.0], [0.5], [1.0]])
    y = np.array([[0.0], [0.25], [1.0]])

    model = RBF().fit(x, y)
    pred = model.predict(np.array([[0.2], [0.8]]))

    assert pred.shape == (2, 1)
    assert np.isfinite(pred).all()


def test_rbf_predict_before_fit_raises_clear_error():
    model = RBF()

    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        model.predict(np.array([[0.1]]))


def test_rbf_kernel_switch_invalidates_fit_state():
    x = np.array([[0.0], [0.5], [1.0]])
    y = np.array([[0.0], [0.25], [1.0]])

    model = RBF(kernel=Cubic()).fit(x, y)
    model.setKernel(Gaussian())

    with pytest.raises(RuntimeError, match="missing fitted state"):
        model.predict(np.array([[0.2]]))


def test_rbf_kernel_choice_apply_parameter_values_switches_kernel():
    model = RBF()
    model.setKernelChoices([Cubic(), Gaussian()])

    model.applyParameterValues(["kernel"], [1.2])

    assert model.kernel.displayName == "Gaussian"
    assert model.setting.get("kernel").displayName == "Gaussian"
