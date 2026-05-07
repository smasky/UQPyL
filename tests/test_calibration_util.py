import numpy as np

from UQPyL.calibration.util import (
    kge,
    mae,
    mse,
    nse,
    pbias,
    pearson_r,
    pfactor,
    r2,
    rfactor,
    rmse,
)


def test_batch_metrics_basic_perfect_and_imperfect_cases():
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],
    ])

    assert np.allclose(mse(obs, sim), [0.0, 1.0 / 3.0])
    assert np.allclose(mae(obs, sim), [0.0, 1.0 / 3.0])
    assert np.allclose(rmse(obs, sim), [0.0, np.sqrt(1.0 / 3.0)])
    assert np.allclose(nse(obs, sim), [1.0, 0.5])
    assert np.allclose(r2(obs, sim), [1.0, 0.5])
    assert np.allclose(pbias(obs, sim), [0.0, 100.0 / 6.0])
    assert np.allclose(pearson_r(obs, sim), [1.0, 0.9819805060619659])
    assert np.allclose(kge(obs, sim)[0], 1.0)


def test_metrics_support_mask():
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([
        [1.0, 9.0, 3.0],
        [1.0, 0.0, 4.0],
    ])
    mask = np.array([False, True, False])

    assert np.allclose(mse(obs, sim, mask=mask), [0.0, 0.5])
    assert np.allclose(mae(obs, sim, mask=mask), [0.0, 0.5])
    assert np.allclose(rmse(obs, sim, mask=mask), [0.0, np.sqrt(0.5)])


def test_pfactor_and_rfactor():
    obs = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.8, 1.9, 2.7])
    upper = np.array([1.2, 2.1, 3.3])

    assert np.allclose(pfactor(obs, lower, upper), 1.0)
    assert np.allclose(rfactor(obs, lower, upper), 0.4898979485566355)
