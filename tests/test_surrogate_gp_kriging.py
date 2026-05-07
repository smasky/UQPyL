import numpy as np
import pytest

from UQPyL.surrogate.gp.gaussian_process import GPR
from UQPyL.surrogate.kriging.kriging import KRG
from UQPyL.surrogate.gp.kernel import RBF as GPRBF, Matern
from UQPyL.optimization.soea.ga import GA
from UQPyL.optimization.base import AlgorithmABC


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

    assert model.setting.get("kernel").displayName == "RBF"

    model.applyParameterValues(["kernel"], [1.2])

    assert model.kernel.displayName == "Matern"
    assert model.setting.get("kernel").displayName == "Matern"


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


def test_gpr_krg_accept_lbfgsb_string_optimizer():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    gpr = GPR(optimizer="LBFGSB")
    krg = KRG(optimizer="LBFGSB")

    assert gpr.fit(x, y) is gpr
    assert krg.fit(x, y) is krg

    assert np.asarray(gpr.predict(x[:2])).shape == (2, 1)
    assert np.asarray(krg.predict(x[:2])).shape == (2, 1)


def test_krg_ea_branch_respects_setting_log_transform():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    model = KRG(optimizer=GA(maxFEs=10, nPop=5), nRestartTimes=0)
    model.fit(x, y)

    theta = np.asarray(model.setting.get("theta")).reshape(-1)
    assert np.all(theta > 0.0)


def test_gpr_krg_default_mp_restart_times_are_five():
    gpr = GPR()
    krg = KRG()

    assert gpr.nRes == 5
    assert krg.nRes == 5


def test_gpr_krg_ea_restart_times_still_follow_explicit_argument():
    gpr = GPR(optimizer=GA(maxFEs=10, nPop=5), nRestartTimes=0)
    krg = KRG(optimizer=GA(maxFEs=10, nPop=5), nRestartTimes=0)

    assert gpr.nRes == 0
    assert krg.nRes == 0


def test_gpr_krg_spawn_deterministic_distinct_child_seeds_for_ea_optimizer():
    class _DummyEA(AlgorithmABC):
        name = "DummyEA"
        alg_type = "EA"

        def __init__(self):
            super().__init__(maxFEs=1, maxIters=1, verboseFlag=False, logFlag=False, saveFlag=False)
            self.seeds = []

        def run(self, problem, seed=None, **kwargs):
            self.seeds.append(seed)

            class _Res:
                bestDecs = np.zeros((1, problem.nInput))
                bestObjs = np.zeros((1, 1))

            return _Res()

    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    gpr_opt = _DummyEA()
    gpr = GPR(optimizer=gpr_opt, nRestartTimes=2)
    gpr.fit(x, y)
    assert len(gpr_opt.seeds) == 3
    assert len(set(gpr_opt.seeds)) == 3

    gpr_opt_1 = _DummyEA()
    gpr1 = GPR(optimizer=gpr_opt_1, nRestartTimes=2)
    gpr1.rng = np.random.default_rng(123)
    gpr1.fit(x, y)

    gpr_opt_2 = _DummyEA()
    gpr2 = GPR(optimizer=gpr_opt_2, nRestartTimes=2)
    gpr2.rng = np.random.default_rng(123)
    gpr2.fit(x, y)
    assert gpr_opt_1.seeds == gpr_opt_2.seeds

    krg_opt = _DummyEA()
    krg = KRG(optimizer=krg_opt, nRestartTimes=2)
    krg.fit(x, y)
    assert len(krg_opt.seeds) == 3
    assert len(set(krg_opt.seeds)) == 3
