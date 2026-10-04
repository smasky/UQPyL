"""Native iteration status must reach Python without writing to stderr."""

import warnings

import numpy as np
import pytest

from UQPyL.surrogate import AutoTuner
from UQPyL.surrogate.svr import SVR


@pytest.mark.parametrize("symbol", ["epsilon-SVR", "nu-SVR"])
def test_limit_warns_and_retains_approximation_then_refit_clears_status(symbol, capfd):
    x = np.linspace(0, 1, 64)[:, None]
    y = np.sin(8 * x)
    model = SVR(symbol=symbol, maxIter=1)
    with pytest.warns(RuntimeWarning, match="SVR reached maxIter=1"):
        model.fit(x, y)
    assert model.fitState["solver"] == {"iterations": 1, "maxIterations": 1, "iterationLimitReached": True}
    assert np.isfinite(model.predict(x)).all()
    assert capfd.readouterr().err == ""
    model.setting.set("maxIter", 100000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    assert not caught
    assert not model.fitState["solver"]["iterationLimitReached"]
    assert 0 <= model.fitState["solver"]["iterations"] < 100000
    assert capfd.readouterr().err == ""


@pytest.mark.parametrize("mode", ["joint", "separate"])
def test_tuner_records_candidate_and_final_solver_status(mode, capfd):
    x = np.linspace(0, 1, 64)[:, None]
    model = SVR(maxIter=1)
    tuner = AutoTuner(model)
    with pytest.warns(RuntimeWarning, match="SVR reached maxIter=1"):
        tuner.gridTune(x, np.sin(8 * x), {"C": np.log([1, 10])}, ratio=25, seed=11, tuneMode=mode)
    report = tuner.getReport()
    for fit in [*(candidate["fit"] for candidate in report["candidates"]), report["final_refit"]]:
        assert fit["status"] == "finished"
        assert fit["solver"] == {"iterations": 1, "max_iterations": 1, "iteration_limit_reached": True}
    assert np.isfinite(model.predict(x)).all()
    assert capfd.readouterr().err == ""


def test_warning_as_error_invalidates_model():
    model = SVR(maxIter=1)
    x = np.linspace(0, 1, 64)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="SVR reached"):
            model.fit(x, np.sin(8 * x))
    with pytest.raises(RuntimeError):
        model.predict(x)
