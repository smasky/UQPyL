import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem


def test_glue_filters_behavioral_samples_by_rmse_threshold():
    obs = np.array([[1.0], [2.0]])

    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 1))
        sim[:, 0, 0] = X[:, 0]
        sim[:, 1, 0] = X[:, 1]
        return sim

    problem = ModelProblem(
        nInput=2,
        ub=3.0,
        lb=0.0,
        simFunc=simf,
        obs=obs,
        simLabels=["Q"],
        name="ToyModel",
    )

    X = np.array([
        [1.0, 2.0],  # perfect
        [1.0, 2.4],  # behavioral
        [0.0, 0.0],  # non-behavioral
    ])

    method = GLUE(verboseFlag=False)
    res = method.run(problem, X, threshold=0.3)

    assert res.method == "GLUE"
    assert np.allclose(res.bestDecs, [[1.0, 2.0]])
    assert np.allclose(res.bestSim, [[1.0, 2.0]])
    assert np.allclose(res.behavioralDecs, [[1.0, 2.0], [1.0, 2.4]])
    assert np.allclose(res.behavioralSims, [[1.0, 2.0], [1.0, 2.4]])
    assert np.allclose(res.diagnostics["scores"], [0.0, 0.2828427124746191, 1.5811388300841898])
    assert np.array_equal(res.diagnostics["behavioralMask"], np.array([True, True, False]))
    assert np.allclose(res.diagnostics["behavioralScores"], [0.0, 0.2828427124746191])


def test_glue_supports_named_metric_configuration():
    obs = np.array([[1.0], [2.0]])

    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 1))
        sim[:, 0, 0] = X[:, 0]
        sim[:, 1, 0] = X[:, 1]
        return sim

    problem = ModelProblem(
        nInput=2,
        ub=3.0,
        lb=0.0,
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

    method = GLUE(verboseFlag=False, metric="nse")
    res = method.run(problem, X, threshold=0.0)

    assert np.allclose(res.diagnostics["scores"], [1.0, -1.0, -9.0])
    assert np.array_equal(res.diagnostics["behavioralMask"], np.array([True, False, False]))
