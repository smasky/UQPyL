import numpy as np

from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.rbf.radial_basis_function import RBF
from UQPyL.util.scaler import StandardScaler


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

