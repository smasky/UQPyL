import numpy as np

from UQPyL.calibration import ES
from UQPyL.problem import ModelProblem


def test_es_updates_ensemble_towards_observation_in_single_pass():
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
        [0.0, 0.0],
        [2.0, 3.0],
        [1.5, 0.5],
    ])

    method = ES(verboseFlag=False)
    res = method.run(problem, X)

    updated = res.posteriorDecs
    assert updated.shape == (3, 2)
    assert res.bestDecs.shape == (1, 2)
    assert res.posteriorSims.shape == (3, 2)
    assert np.allclose(res.diagnostics["priorMean"], np.array([1.16666667, 1.16666667]))
    assert np.mean(res.diagnostics["scores"]) < np.mean(res.diagnostics["priorScores"])
    assert not np.allclose(updated, X)


def test_es_supports_custom_metric_callable():
    obs = np.array([[1.0], [2.0]])

    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 1))
        sim[:, 0, 0] = X[:, 0]
        sim[:, 1, 0] = X[:, 1]
        return sim

    def mean_bias(obs_vec, sim_mat, mask=None):
        if mask is not None:
            valid = ~mask
            obs_vec = obs_vec[valid]
            sim_mat = sim_mat[:, valid]
        return np.mean(sim_mat - obs_vec.reshape(1, -1), axis=1)

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
        [0.0, 0.0],
        [2.0, 3.0],
        [1.5, 0.5],
    ])

    method = ES(verboseFlag=False, metric=mean_bias)
    res = method.run(problem, X)

    assert "scores" in res.diagnostics
    assert res.diagnostics["scores"].shape == (3,)
    assert res.posteriorDecs.shape == (3, 2)
