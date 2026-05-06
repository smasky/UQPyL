import numpy as np
import pytest

from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.rbf.radial_basis_function import RBF
from UQPyL.surrogate.rbf.kernel import Cubic, Gaussian
from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.util.scaler import StandardScaler
from UQPyL.util.poly import PolyFeature


def test_autotuner_grid_tune_smoke():
    # gridTune doesn't need an optimizer; it brute-forces the grid.
    x = np.linspace(0, 1, 10).reshape(-1, 1)
    y = (x**2) + 0.1

    model = RBF(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)))
    tuner = AutoTuner(model=model, optimizer=None)

    # Tune smoothing parameter across a tiny grid
    best_vals, best_obj = tuner.gridTune(
        xData=x,
        yData=y,
        paraGrid={"C_smooth": [0.0, 1e-6]},
        # Ensure at least 2 test samples so r_square is well-defined (SST > 0).
        ratio=20,
    )
    assert isinstance(best_vals, (float, int, np.ndarray))
    assert np.isfinite(best_obj)


def test_autotuner_grid_tune_with_owner_filter():
    x = np.linspace(0, 1, 10).reshape(-1, 1)
    y = (x**2) + 0.1

    model = RBF(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)))
    tuner = AutoTuner(model=model, optimizer=None)

    best_vals, best_obj = tuner.gridTune(
        xData=x,
        yData=y,
        paraGrid={"C_smooth": [0.0, 1e-6]},
        ratio=20,
        owner="model",
    )
    assert isinstance(best_vals, (float, int, np.ndarray))
    assert np.isfinite(best_obj)


def test_setting_owner_and_autotuner_default_owner_resolution():
    x = np.linspace(0, 1, 10).reshape(-1, 1)
    y = (x**2) + 0.1

    model = RBF(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)))
    tuner = AutoTuner(model=model, optimizer=None)

    assert "C_smooth" in model.setting.getParaList(owner="model")

    model.kernel.initialize(x.shape[1])
    assert "epsilon" in model.setting.getParaList(owner="kernel")

    best_vals, best_obj = tuner.gridTune(
        xData=x,
        yData=y,
        paraGrid=None,
        ratio=20,
        owner="kernel",
    )
    assert isinstance(best_vals, (float, int, np.ndarray))
    assert np.isfinite(best_obj)


def test_autotuner_grid_tune_joint_mode_with_kernel_choice():
    x = np.linspace(0, 1, 10).reshape(-1, 1)
    y = (x**2) + 0.1

    model = RBF(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)))
    model.setKernelChoices([Cubic(), Gaussian()])
    tuner = AutoTuner(model=model, optimizer=None)

    best_vals, best_obj = tuner.gridTune(
        xData=x,
        yData=y,
        paraGrid={"kernel": [0.2, 1.2], "C_smooth": [0.0, 1e-6]},
        ratio=20,
        tuneMode="joint",
    )

    kernel, smooth = best_vals
    assert kernel.displayName in {"Cubic", "Gaussian"}
    assert np.isfinite(float(np.asarray(smooth).reshape(-1)[0]))
    assert np.isfinite(best_obj)


def test_autotuner_joint_mode_skips_fit_hyper():
    class TrackingRBF(RBF):
        def __init__(self):
            super().__init__()
            self.fitModelCalls = 0
            self.fitHyperCalls = 0

        def fitModel(self, xTrain, yTrain):
            self.fitModelCalls += 1
            return super().fitModel(xTrain, yTrain)

        def fitHyper(self, xTrain, yTrain):
            self.fitHyperCalls += 1
            return super().fitHyper(xTrain, yTrain)

    x = np.linspace(0, 1, 10).reshape(-1, 1)
    y = (x**2) + 0.1

    model = TrackingRBF()
    tuner = AutoTuner(model=model, optimizer=None)

    tuner.gridTune(
        xData=x,
        yData=y,
        paraGrid={"C_smooth": [0.0, 1e-6]},
        ratio=20,
        tuneMode="joint",
    )

    assert model.fitModelCalls > 0
    assert model.fitHyperCalls == 0


def test_autotuner_grid_tune_uses_raw_test_data_for_predict():
    x = np.linspace(0, 1, 12).reshape(-1, 1)
    y = 1.0 + 2.0 * x + 0.5 * x**2

    model = LinearRegression(
        scalers=(StandardScaler(0, 1), StandardScaler(0, 1)),
        polyFeature=PolyFeature(degree=2),
        lossType="Ridge",
    )
    tuner = AutoTuner(model=model, optimizer=None)

    best_vals, best_obj = tuner.gridTune(
        xData=x,
        yData=y,
        paraGrid={"C": [1e-4, 1e-2, 1.0]},
        ratio=25,
        tuneMode="joint",
    )

    assert isinstance(best_vals, (float, int, np.ndarray))
    assert np.isfinite(best_obj)

