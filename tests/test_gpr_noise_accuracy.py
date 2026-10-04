"""Noise search must denoise independent points and preserve explicit choices."""

import numpy as np
import pytest
from scipy.stats import qmc

from UQPyL.surrogate import StandardScaler
from UQPyL.surrogate.gp import GPR


@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.parametrize("noiseStd", [0.03, 0.1, 0.3])
@pytest.mark.parametrize("scaled", [False, True])
def test_default_gp_learns_noise_and_predicts_clean_function(seed, noiseStd, scaled):
    trainX = qmc.LatinHypercube(1, seed=seed).random(192)
    trainY = np.sin(2 * np.pi * trainX) + np.random.default_rng(seed + 1000).normal(0, noiseStd, trainX.shape)
    testX = np.linspace(0, 1, 1001)[:, None]
    model = GPR(scalers=(None, StandardScaler() if scaled else None))
    model.rng = np.random.default_rng(seed)
    model.fit(trainX, trainY)
    residual = model.predict(testX) - np.sin(2 * np.pi * testX)
    assert np.sqrt(np.mean(residual**2)) < noiseStd / 2
    expectedVariance = noiseStd**2 / np.var(trainY, ddof=1) if scaled else noiseStd**2
    fittedVariance = float(np.asarray(model.setting.get("C")).item())
    assert 0.4 * expectedVariance < fittedVariance < 2 * expectedVariance


@pytest.mark.parametrize("noiseVariance", [0.0, 0.01])
def test_fixed_noise_is_preserved_and_matches_dense_posterior(noiseVariance):
    x = np.linspace(0, 1, 12)[:, None]
    y = np.sin(6 * x)
    model = GPR(C=noiseVariance, C_attr=None)
    # Fix the kernel too, so a dense solve independently checks the noise term.
    from UQPyL.surrogate.gp.kernel import RBF

    model.setKernel(RBF(length_scale=0.2, length_attr=None))
    model.fit(x, y)
    query = np.array([[0.15], [0.55], [0.85]])
    covariance = np.exp(-0.5 * ((x - x.T) / 0.2) ** 2) + noiseVariance * np.eye(len(x))
    cross = np.exp(-0.5 * ((query - x.T) / 0.2) ** 2)
    np.testing.assert_allclose(model.predict(query), cross @ np.linalg.solve(covariance, y), atol=1e-9)
    assert model.setting.get("C") == noiseVariance


def test_explicit_noise_range_remains_authoritative():
    x = np.linspace(0, 1, 48)[:, None]
    y = np.sin(6 * x) + np.random.default_rng(19).normal(0, 0.1, x.shape)
    model = GPR(C_attr={"lb": 1e-12, "ub": 1e-6, "type": "float", "log": True})
    model.rng = np.random.default_rng(11)
    model.fit(x, y)
    assert 1e-12 * (1 - 1e-12) <= model.setting.get("C") <= 1e-6 * (1 + 1e-12)
