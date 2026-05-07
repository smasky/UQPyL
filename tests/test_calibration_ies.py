import numpy as np

from UQPyL.calibration import ES, IES
from UQPyL.problem import ModelProblem


def test_ies_iterative_updates_reduce_mean_score_beyond_single_es_step():
    obs = np.array([[1.0], [2.0]])

    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 1))
        sim[:, 0, 0] = X[:, 0]
        sim[:, 1, 0] = X[:, 1] ** 2
        return sim

    problem = ModelProblem(
        nInput=2,
        ub=3.0,
        lb=0.0,
        simFunc=simf,
        obs=obs,
        simLabels=["Q"],
        name="NonlinearToyModel",
    )

    X = np.array([
        [0.0, 0.5],
        [2.0, 1.0],
        [1.5, 2.0],
    ])

    es = ES(verboseFlag=False)
    es_res = es.run(problem, X)

    ies = IES(verboseFlag=False, maxIters=4, lam=1e-6)
    ies_res = ies.run(problem, X)

    assert ies_res.method == "IES"
    assert ies_res.bestDecs.shape == (1, 2)
    assert ies_res.posteriorDecs.shape == X.shape
    assert ies_res.posteriorSims.shape == (3, 2)
    assert np.mean(ies_res.diagnostics["scores"]) < np.mean(es_res.diagnostics["scores"])
    assert len(ies_res.history.metricsHistory) == 4
