"""Default zero-noise SVD versus the retained dense calibration path."""

import numpy as np
import pytest

from UQPyL.calibration import ES, IES
from UQPyL.calibration.methods import _ensemble
from UQPyL.calibration.methods._ensemble import anomalyGain
from UQPyL.problem import ModelProblem


@pytest.mark.parametrize("nEns,nObs", [(4, 20), (20, 4), (8, 8)])
@pytest.mark.parametrize("lam", [0.0, 0.1, 2.0])
def testZeroNoiseMatchesDenseGain(nEns, nObs, lam):
    rng = np.random.default_rng(28)
    x, y = rng.normal(size=(nEns, 3)), rng.normal(size=(nEns, nObs))
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    beforeX, beforeY = x.copy(), y.copy()
    gain, info = anomalyGain(x, y, lam=lam)
    reference, denseInfo = anomalyGain(x, y, np.zeros((nObs, nObs)), lam)
    np.testing.assert_allclose(gain, reference, rtol=2e-10, atol=2e-12)
    assert info["rank"] == denseInfo["rank"]
    assert info["solver"] == ("svd" if nObs > nEns else denseInfo["solver"])
    assert info["cutoff"] == pytest.approx(denseInfo["cutoff"], rel=1e-12, abs=0)
    np.testing.assert_array_equal(x, beforeX)
    np.testing.assert_array_equal(y, beforeY)


@pytest.mark.parametrize("lam,rank", [(0.0, 0), (0.1, 200)])
def testZeroSpreadAndRidgeNullspaceRank(lam, rank):
    gain, info = anomalyGain(np.zeros((4, 2)), np.zeros((4, 200)), lam=lam)
    np.testing.assert_array_equal(gain, np.zeros((2, 200)))
    assert info["rank"] == rank


def testObservationDimensionControlsNumericalRank():
    # Orthogonal centered columns with eigenvalues 1, 1e-12, 1e-18.
    u = np.zeros((6, 3))
    for index in range(3):
        u[2 * index : 2 * index + 2, index] = [1, -1]
    u /= np.sqrt(2)
    y = np.zeros((6, 20))
    y[:, :3] = u * np.array([1, 1e-6, 1e-9]) * np.sqrt(5)
    x = u * np.sqrt(5)
    gain, info = anomalyGain(x, y)
    expected = np.zeros((3, 20))
    expected[0, 0], expected[1, 1] = 1, 1e6
    np.testing.assert_allclose(gain, expected, atol=1e-10, rtol=1e-12)
    assert info["rank"] == 2
    assert info["cutoff"] == pytest.approx(20 * np.finfo(float).eps, abs=0)


@pytest.mark.parametrize("methodClass,lam", [(ES, 0.0), (IES, 0.0), (IES, 0.1)])
def testRealRunMatchesExplicitZeroNoiseWithMaskAndBounds(methodClass, lam):
    calls = []
    transform = np.array([[1.0, 2.0, -1.0, 4.0, 0.0, 3.0], [0.0, 1.0, 2.0, -1.0, 3.0, 1.0]])

    def simulate(x):
        assert np.all((x >= 0) & (x <= 1))
        calls.append(x.copy())
        return x @ transform

    mask = np.array([[False], [True], [False], [False], [True], [False]])
    problem = ModelProblem(nInput=2, lb=0.0, ub=1.0, simFunc=simulate, obs=np.full(6, 10.0), mask=None if mask is None else mask.reshape(-1))
    x = np.random.default_rng(3).uniform(0.1, 0.9, (8, 2))
    options = {"maxIters": 3, "lam": lam} if methodClass is IES else {}
    first = methodClass(**options).run(problem, x)
    count = len(calls)
    second = methodClass(**options).run(problem, x, r=np.zeros((4, 4)))
    assert len(calls) == 2 * count and count == (4 if methodClass is IES else 2)
    np.testing.assert_allclose(first.posteriorDecs, second.posteriorDecs, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(first.posteriorSims, second.posteriorSims, atol=1e-11, rtol=1e-10)
    assert first.diagnostics["boundUpdates"] == second.diagnostics["boundUpdates"]


@pytest.mark.parametrize("methodClass", [ES, IES])
def testDefaultRunNeverBuildsDenseCovariance(methodClass, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Default zero noise must not construct or factor a dense covariance.")

    monkeypatch.setattr(_ensemble, "ensembleGain", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    import UQPyL.calibration.methods.es as esModule

    monkeypatch.setattr(esModule, "validateCovariance", forbidden)
    svd = np.linalg.svd
    shapes = []

    def tracked(a, **kwargs):
        shapes.append(a.shape)
        assert kwargs["full_matrices"] is False
        return svd(a, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", tracked)
    problem = ModelProblem(
        nInput=1, lb=-10.0, ub=10.0, obs=np.ones(400), simFunc=lambda x: (np.repeat(x[:, None, :], 400, axis=1)).reshape(len(x), -1)
    )
    options = {"maxIters": 1, "lam": 0.1} if methodClass is IES else {}
    result = methodClass(**options).run(problem, [[-1.0], [0.0], [2.0]])
    assert shapes == ([(3, 1), (3, 400)] if methodClass is IES else [(3, 400)])
    assert np.all(np.isfinite(result.posteriorDecs))
