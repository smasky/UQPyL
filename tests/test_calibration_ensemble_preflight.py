"""Calibration ensemble preflight.

Migrated from test_review_c06_c12.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from UQPyL.calibration import ES, IES
from UQPyL.problem import Problem, ModelProblem


def makeModelProblem(calls, obs=None, mask=None):
    def simulate(x):
        calls.append(x.copy())
        return (np.repeat(x[:, None, :], 3, axis=1)).reshape(len(x), -1)

    return ModelProblem(nInput=1, lb=-5, ub=5, simFunc=simulate, obs=(np.ones((3, 1)) if obs is None else obs).reshape(-1), mask=None if mask is None else mask.reshape(-1))


# Regression source: test_review_c06_c12.py::testEnsembleRejectsInvalidCovarianceBeforeSimulation
@pytest.mark.parametrize("methodClass", [ES, IES])
@pytest.mark.parametrize("covariance", [np.eye(2), np.full((3, 3), np.nan), np.triu(np.ones((3, 3))), -np.eye(3)])
def testEnsembleRejectsInvalidCovarianceBeforeSimulation(methodClass, covariance):
    calls = []
    with pytest.raises(ValueError, match="covariance"):
        methodClass().run(makeModelProblem(calls), [[-1], [0], [2]], r=covariance)
    assert calls == []


# Regression source: test_review_c06_c12.py::testIesValidatesRuntimeSettingsBeforeSimulation
@pytest.mark.parametrize(
    "key,value",
    [
        ("lam", -1),
        ("lam", np.nan),
        ("lam", [1, 2]),
        ("maxIters", -1),
        ("maxIters", 1.5),
        ("maxIters", True),
        ("maxIters", None),
    ],
)
def testIesValidatesRuntimeSettingsBeforeSimulation(key, value):
    method = IES()
    method.set(key, value)
    calls = []
    with pytest.raises(ValueError, match=key):
        method.run(makeModelProblem(calls), [[-1], [0], [2]])
    assert calls == []


# Regression source: test_review_c06_c12.py::testEnsembleRejectsInvalidObservationsBeforeSimulation
@pytest.mark.parametrize("methodClass", [ES, IES])
@pytest.mark.parametrize("kind", ["masked", "infinite"])
def testEnsembleRejectsInvalidObservationsBeforeSimulation(methodClass, kind):
    calls = []
    obs = np.ones((3, 1))
    mask = np.ones((3, 1), dtype=bool) if kind == "masked" else None
    if kind == "infinite":
        obs[0, 0] = np.inf
    with pytest.raises(ValueError, match="observation"):
        methodClass().run(makeModelProblem(calls, obs, mask), [[-1], [0], [2]])
    assert calls == []


# Regression source: test_review_c06_c12.py::testIesValidatesCovarianceOnceAndReusesSimulation
@pytest.mark.parametrize("maxIters", [0, 1, 3])
def testIesValidatesCovarianceOnceAndReusesSimulation(maxIters, monkeypatch):
    import UQPyL.calibration.methods.es as esModule

    original = esModule.validateCovariance
    validations = []

    def validate(r, nObs):
        validations.append(nObs)
        return original(r, nObs)

    monkeypatch.setattr(esModule, "validateCovariance", validate)
    calls = []
    problem = makeModelProblem(calls, mask=np.array([[False], [True], [False]]))
    result = IES(maxIters=maxIters).run(problem, [[-1], [0], [2]], r=np.eye(2))
    assert validations == [2]
    assert len(calls) == maxIters + 1
    assert result.posteriorSims.shape == (3, 3)


# Regression source: test_review_c06_c12.py::testIesSingleUpdateAlsoValidatesBeforeSimulation
def testIesSingleUpdateAlsoValidatesBeforeSimulation():
    calls = []
    method = IES()
    method.setup(makeModelProblem(calls))
    with pytest.raises(ValueError, match="covariance"):
        method._update_once([[-1], [0], [2]], r=np.eye(2))
    with pytest.raises(ValueError, match="lam"):
        method._update_once([[-1], [0], [2]], lam=-1)
    assert calls == []
