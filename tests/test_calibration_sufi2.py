import numpy as np

from UQPyL.calibration import SUFI2
from UQPyL.problem import ModelProblem


def test_sufi2_selects_elite_samples_and_updates_bounds_with_95ppu_metrics():
    obs = np.array([[1.0], [2.0]])

    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 1))
        sim[:, 0, 0] = X[:, 0]
        sim[:, 1, 0] = X[:, 1]
        return sim

    problem = ModelProblem(
        nInput=2,
        ub=[3.0, 3.0],
        lb=[0.0, 0.0],
        simFunc=simf,
        obs=obs,
        simLabels=["Q"],
        name="ToyModel",
    )

    X = np.array([
        [1.0, 2.0],
        [1.2, 2.2],
        [0.0, 0.0],
    ])

    method = SUFI2(verboseFlag=False)
    res = method.run(problem, X, eliteSize=2)

    assert res.method == "SUFI2"
    assert np.allclose(res.bestDecs, [[1.0, 2.0]])
    assert np.allclose(res.bestSim, [[1.0, 2.0]])
    assert np.allclose(res.eliteDecs, [[1.0, 2.0], [1.2, 2.2]])
    assert np.allclose(res.eliteSims, [[1.0, 2.0], [1.2, 2.2]])
    assert np.allclose(res.diagnostics["scores"], [0.0, 0.2, 1.5811388300841898])
    assert np.array_equal(res.diagnostics["eliteMask"], np.array([True, True, False]))
    assert np.allclose(res.diagnostics["updatedLb"], [1.0, 2.0])
    assert np.allclose(res.diagnostics["updatedUb"], [1.2, 2.2])
    assert np.allclose(res.diagnostics["pfactor"], 0.0)
    assert np.allclose(res.diagnostics["rfactor"], 0.38)
    assert np.allclose(res.diagnostics["ppuLower"], [1.005, 2.005])
    assert np.allclose(res.diagnostics["ppuUpper"], [1.195, 2.195])


def test_sufi2_supports_named_metric_configuration():
    obs = np.array([[1.0], [2.0]])

    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 1))
        sim[:, 0, 0] = X[:, 0]
        sim[:, 1, 0] = X[:, 1]
        return sim

    problem = ModelProblem(
        nInput=2,
        ub=[3.0, 3.0],
        lb=[0.0, 0.0],
        simFunc=simf,
        obs=obs,
        simLabels=["Q"],
        name="ToyModel",
    )

    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [0.0, 0.0],
    ])

    method = SUFI2(verboseFlag=False, metric="nse")
    res = method.run(problem, X, eliteSize=2)

    assert np.allclose(res.diagnostics["scores"], [1.0, -1.0, -9.0])
    assert np.array_equal(res.diagnostics["eliteMask"], np.array([True, True, False]))


def test_sufi2_iterative_mode_samples_and_updates_bounds_history():
    obs = np.array([[1.0], [2.0]])

    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 1))
        sim[:, 0, 0] = X[:, 0]
        sim[:, 1, 0] = X[:, 1]
        return sim

    problem = ModelProblem(
        nInput=2,
        ub=[3.0, 3.0],
        lb=[0.0, 0.0],
        simFunc=simf,
        obs=obs,
        simLabels=["Q"],
        name="ToyModel",
    )

    method = SUFI2(verboseFlag=False, maxIters=3, nSamples=12)
    res = method.run(problem, eliteSize=4, seed=123)

    assert res.bestDecs.shape == (1, 2)
    assert res.posteriorDecs.shape[1] == 2
    assert res.posteriorSims.shape[1] == 2
    assert len(res.history.metricsHistory) == 3
    first = res.history.metricsHistory[0]
    last = res.history.metricsHistory[-1]
    assert "updatedLb" in first and "updatedUb" in first
    assert np.all(np.asarray(last["updatedUb"]) - np.asarray(last["updatedLb"]) <= np.array([3.0, 3.0]))
