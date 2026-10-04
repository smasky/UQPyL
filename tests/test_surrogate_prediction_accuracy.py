"""Independent predictions detect restrictive defaults and validate tuning."""

import numpy as np
import pytest
from scipy.stats import qmc

from UQPyL.surrogate import AutoTuner
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import Matern, RBF, RationalQuadratic
from UQPyL.surrogate.mars import MARS
from UQPyL.surrogate.svr import SVR


def predictionR2(truth, prediction):
    return 1 - np.sum((truth - prediction) ** 2) / np.sum((truth - truth.mean()) ** 2)


@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.parametrize("kernelClass", [RBF, Matern, RationalQuadratic])
@pytest.mark.numerical
def test_gp_default_length_domain_resolves_short_scale_function(seed, kernelClass):
    trainX = qmc.LatinHypercube(1, seed=seed).random(128)
    testX = np.linspace(0, 1, 1001)[:, None]
    model = GPR(kernel=kernelClass())
    model.rng = np.random.default_rng(seed)
    model.fit(trainX, np.sin(8 * np.pi * trainX))
    prediction = model.predict(testX)
    assert predictionR2(np.sin(8 * np.pi * testX), prediction) > 0.995


@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.numerical
def test_default_mars_recovers_pairwise_interaction(seed):
    trainX = qmc.LatinHypercube(2, seed=seed).random(64)
    testX = np.random.default_rng(712).uniform(size=(1024, 2))
    trainY = np.prod(2 * trainX - 1, axis=1)[:, None]
    testY = np.prod(2 * testX - 1, axis=1)[:, None]
    model = MARS().fit(trainX, trainY)
    np.testing.assert_allclose(model.predict(testX), testY, atol=1e-10)
    additive = MARS(max_degree=1).fit(trainX, trainY)
    assert predictionR2(testY, additive.predict(testX)) < 0.2


@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.numerical
def test_svr_validation_tuning_generalizes_to_independent_oscillatory_points(seed):
    trainX = qmc.LatinHypercube(1, seed=seed).random(128)
    trainY = np.sin(8 * np.pi * trainX)
    testX = np.linspace(0, 1, 1001)[:, None]
    testY = np.sin(8 * np.pi * testX)
    model = SVR()
    tuner = AutoTuner(model)
    _, score = tuner.gridTune(
        trainX,
        trainY,
        paraGrid={"C": np.log([1, 100]), "epsilon": np.log([0.001, 0.01]), "gamma": np.log([1, 10, 100])},
        ratio=25,
        seed=seed,
        tuneMode="joint",
    )
    assert score > 0.99
    assert predictionR2(testY, model.predict(testX)) > 0.99
    assert len(model.xTrain) == len(trainX)


@pytest.mark.numerical
def test_gp_default_uncertainty_no_longer_misses_smooth_function_systematically():
    trainX = qmc.LatinHypercube(2, seed=23).random(128)
    testX = np.random.default_rng(712).uniform(size=(4096, 2))

    def function(x):
        return (np.sin(2 * np.pi * x[:, 0]) + 0.5 * np.cos(2 * np.pi * x[:, 1]))[:, None]

    model = GPR()
    model.rng = np.random.default_rng(23)
    model.fit(trainX, function(trainX))
    prediction, std = model.predict(testX, returnStd=True)
    assert predictionR2(function(testX), prediction) > 0.99999
    # A fixed misspecification regression, not a universal calibration claim.
    assert np.mean(np.abs(prediction - function(testX)) <= 1.96 * std) > 0.9
