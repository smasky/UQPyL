"""Independent high-precision RMSE and public screening across output units."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from UQPyL.calibration import GLUE, SUFI2
from UQPyL.calibration.util import rmse
from UQPyL.problem import ModelProblem


def decimalRmse(obs, sim):
    with localcontext() as context:
        context.prec = 100
        squared = [(Decimal(float(a)) - Decimal(float(b))) ** 2 for a, b in zip(obs, sim)]
        return float((sum(squared) / Decimal(len(squared))).sqrt())


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e-12, 1.0, 1e150, 1e200, 1e300])
@pytest.mark.parametrize("masked", [False, True])
def test_rmse_matches_decimal_with_independent_row_scaling(scale, masked):
    obs = np.array([1.0, 2.0, 3.0]) * scale
    sim = np.array([[1.1, 1.8, 3.05], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]) * scale
    expected = [decimalRmse(obs, row) for row in sim]
    if masked:
        obs = np.append(obs, np.nan)
        sim = np.column_stack([sim, [np.nan] * 3])
    beforeObs, beforeSim = obs.copy(), sim.copy()
    actual = rmse(obs, sim, mask=[False, False, False, True] if masked else None)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(obs, beforeObs)
    np.testing.assert_array_equal(sim, beforeSim)


@pytest.mark.parametrize(
    "obs,sim",
    [
        ([1e308, 0.0], [1e308, 1e-200]),
        ([-1e308, 0.0, 0.0, 0.0], [1e308, 0.0, 0.0, 0.0]),
        ([0.0], [np.nextafter(0.0, 1.0)]),
        ([1e300], [np.nextafter(1e300, np.inf)]),
    ],
)
def test_rmse_residual_edge_cases(obs, sim):
    np.testing.assert_allclose(rmse(obs, sim), [decimalRmse(obs, sim)], rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "obs,sim,expected",
    [
        ([-1e308], [1e308], np.inf),
        ([0.0] * 16, [np.nextafter(0.0, 1.0)] + [0.0] * 15, 0.0),
    ],
)
def test_unrepresentable_final_rmse_warns(obs, sim, expected):
    with pytest.warns(RuntimeWarning, match="RMSE exceeds floating-point range"):
        assert rmse(obs, sim)[0] == expected


@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e200])
@pytest.mark.parametrize("kind", ["GLUE", "SUFI2"])
def test_scale_does_not_change_actual_calibration_selection(scale, kind):
    obs = np.array([1.0, 2.0]) * scale
    simulations = np.array([[1.1, 2.2], [1.0, 2.1]]) * scale
    problem = ModelProblem(
        nInput=1, lb=0, ub=1, obs=(obs[:, None]).reshape(-1), simFunc=lambda x: (simulations[np.asarray(x[:, 0], dtype=int), :, None]).reshape(len(x), -1)
    )
    x = np.array([[0.0], [1.0]])
    result = GLUE().run(problem, x, threshold=0.1 * scale) if kind == "GLUE" else SUFI2().run(problem, x, eliteSize=1)
    np.testing.assert_array_equal(result.bestDecs, [[1.0]])
    if kind == "GLUE":
        np.testing.assert_array_equal(result.diagnostics["behavioralMask"], [False, True])
