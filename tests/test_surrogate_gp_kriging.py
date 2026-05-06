import numpy as np
import pytest

from UQPyL.surrogate.gp.gaussian_process import GPR
from UQPyL.surrogate.kriging.kriging import KRG
from UQPyL.surrogate.gp.kernel import RBF as GPRBF, Matern
from UQPyL.optimization.soea.ga import GA


def test_gpr_predict_protocol_smoke():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    model = GPR()
    assert model.fit(x, y) is model

    mean = model.predict(x[:3])
    mean_std, std = model.predict(x[:3], returnStd=True)
    mean_var, var = model.predict(x[:3], returnVar=True)

    assert np.asarray(mean).shape == (3, 1)
    assert np.asarray(mean_std).shape == (3, 1)
    assert np.asarray(std).shape == (3, 1)
    assert np.asarray(mean_var).shape == (3, 1)
    assert np.asarray(var).shape == (3, 1)


def test_krg_predict_protocol_smoke():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    model = KRG()
    assert model.fit(x, y) is model

    mean = model.predict(x[:3])
    mean_std, std = model.predict(x[:3], returnStd=True)
    mean_var, var = model.predict(x[:3], returnVar=True)

    assert np.asarray(mean).shape == (3, 1)
    assert np.asarray(mean_std).shape == (3, 1)
    assert np.asarray(std).shape == (3, 1)
    assert np.asarray(mean_var).shape == (3, 1)
    assert np.asarray(var).shape == (3, 1)


def test_gpr_predict_before_fit_raises_clear_error():
    model = GPR()

    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        model.predict(np.array([[0.1]]))


def test_krg_predict_before_fit_raises_clear_error():
    model = KRG()

    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        model.predict(np.array([[0.1]]))


def test_gpr_kernel_choice_apply_parameter_values_switches_kernel():
    model = GPR()
    model.setKernelChoices([GPRBF(), Matern()])

    assert model.setting.getVals("kernel").displayName == "RBF"

    model.applyParameterValues(["kernel"], [1.2])

    assert model.kernel.displayName == "Matern"
    assert model.setting.getVals("kernel").displayName == "Matern"


def test_gpr_predict_missing_fit_state_raises_clear_error():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    model = GPR().fit(x, y)
    model.fitState.pop("alpha")

    with pytest.raises(RuntimeError, match="missing fitted state: alpha"):
        model.predict(x[:1])


def test_gpr_kernel_switch_invalidates_fit_state():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    model = GPR().fit(x, y)
    model.setKernel(Matern())

    with pytest.raises(RuntimeError, match="missing fitted state"):
        model.predict(x[:1])


def test_krg_kernel_switch_invalidates_fit_state():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    model = KRG().fit(x, y)
    model.setKernel(model.kernel.__class__())

    with pytest.raises(RuntimeError, match="missing fitted state"):
        model.predict(x[:1])


def test_gpr_and_krg_record_fit_objective():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    gpr = GPR().fit(x, y)
    krg = KRG().fit(x, y)

    assert np.isfinite(float(np.asarray(gpr.fitState["objective"]).reshape(-1)[0]))
    assert np.isfinite(float(np.asarray(krg.fitState["objective"]).reshape(-1)[0]))


def test_gpr_krg_only_accept_boxmin_string_optimizer():
    with pytest.raises(ValueError, match="optimizer must be 'Boxmin'"):
        GPR(optimizer="GA")

    with pytest.raises(ValueError, match="optimizer must be 'Boxmin'"):
        KRG(optimizer="GA")


def test_krg_ea_branch_respects_setting_log_transform():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    model = KRG(optimizer=GA(maxFEs=10, nPop=5), nRestartTimes=0)
    model.fit(x, y)

    theta = np.asarray(model.setting.getVals("theta")).reshape(-1)
    assert np.all(theta > 0.0)
