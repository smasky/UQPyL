from types import SimpleNamespace

import numpy as np
import pytest

from UQPyL.optimization.base import AlgorithmABC
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import Matern, RBF, RationalQuadratic
from UQPyL.surrogate.scaler import StandardScaler


def referenceKernel(X, Z, lengthScale, family):
    distanceSquared = np.sum(((X[:, None, :] - Z[None, :, :]) / lengthScale)**2, axis=2)
    if family == "rbf":
        return np.exp(-0.5 * distanceSquared)
    if family == "matern":
        radius = np.sqrt(5 * distanceSquared)
        return (1 + radius + radius**2 / 3) * np.exp(-radius)
    return (1 + distanceSquared / (2 * 0.7))**-0.7


def referenceObjective(covariance, Y):
    # Independent dense solve/log-determinant, without GPR's Cholesky factors.
    sign, logDet = np.linalg.slogdet(covariance)
    assert sign == 1
    return 0.5 * (np.sum(Y * np.linalg.solve(covariance, Y))
                  + Y.shape[1] * (logDet + len(Y) * np.log(2 * np.pi)))


@pytest.mark.parametrize("family", ["rbf", "matern", "rq"])
@pytest.mark.parametrize("nInputs,nOutputs", [(1, 1), (1, 2), (3, 1), (3, 2)])
@pytest.mark.parametrize("scaled", [False, True])
def test_fixed_parameters_match_independent_objective_and_posterior(family, nInputs, nOutputs, scaled):
    rng = np.random.default_rng(39)
    X = rng.uniform(-2, 2, (9, nInputs))
    XTest = rng.uniform(-2, 2, (5, nInputs))
    Y = rng.normal(size=(9, nOutputs)) + np.arange(nOutputs) + 2
    lengthScale = np.linspace(0.4, 1.2, nInputs)
    kernelArgs = dict(length_scale=lengthScale, length_attr=None, heterogeneous=True)
    if family == "rbf":
        kernel = RBF(**kernelArgs)
    elif family == "matern":
        kernel = Matern(**kernelArgs, nu=2.5)
    else:
        kernel = RationalQuadratic(**kernelArgs, alpha=0.7, alpha_attr=None)
    scalers = (StandardScaler(), StandardScaler()) if scaled else (None, None)
    model = GPR(kernel=kernel, C=0.02, C_attr=None, scalers=scalers).fit(X, Y)

    xOffset, xScale = (X.mean(0), X.std(0, ddof=1)) if scaled else (0, 1)
    yOffset, yScale = (Y.mean(0), Y.std(0, ddof=1)) if scaled else (0, 1)
    xPrepared, yPrepared = (X - xOffset) / xScale, (Y - yOffset) / yScale
    xTestPrepared = (XTest - xOffset) / xScale
    covariance = referenceKernel(xPrepared, xPrepared, lengthScale, family) + 0.02 * np.eye(len(X))
    crossCovariance = referenceKernel(xTestPrepared, xPrepared, lengthScale, family)
    expectedMean = crossCovariance @ np.linalg.solve(covariance, yPrepared) * yScale + yOffset
    expectedVar = 1 - np.sum(crossCovariance * np.linalg.solve(covariance, crossCovariance.T).T, axis=1)
    expectedVar = np.broadcast_to(expectedVar[:, None] * yScale**2, (len(XTest), nOutputs))

    assert model.fitState["objective"] == pytest.approx(referenceObjective(covariance, yPrepared), rel=1e-10)
    mean, variance = model.predict(XTest, returnVar=True)
    meanWithStd, std = model.predict(XTest, returnStd=True)
    np.testing.assert_allclose(mean, expectedMean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(variance, expectedVar, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(meanWithStd, expectedMean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(std**2, expectedVar, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("case", ["single", "zero", "duplicate", "jitter"])
def test_objective_matches_effective_covariance_for_degenerate_data(case):
    X = np.array([[0.0], [0.0], [1.0]])
    Y = np.array([[1.0], [1.0], [0.2]])
    noise = 0.01
    if case == "single":
        X, Y = X[:1], Y[:1]
    elif case == "zero":
        Y = np.zeros_like(Y)
    elif case == "jitter":
        noise = 0.0
    model = GPR(kernel=RBF(length_scale=0.5, length_attr=None), C=noise, C_attr=None).fit(X, Y)
    effectiveNoise = 1e-6 if case == "jitter" else noise
    covariance = referenceKernel(X, X, 0.5, "rbf") + effectiveNoise * np.eye(len(X))
    assert model.fitState["objective"] == pytest.approx(referenceObjective(covariance, Y), rel=1e-8)
    np.testing.assert_allclose(model.fitState["L"] @ model.fitState["L"].T, covariance, atol=1e-12)
    assert np.all(np.isfinite(model.predict(X)))


@pytest.mark.parametrize("lengths", [[0.3, 0.36, 3.0], [3.0, 0.3, 0.36]])
def test_ea_restarts_keep_best_likelihood_and_rebuild_its_state(lengths):
    class CandidateEA(AlgorithmABC):
        name = "CandidateEA"
        alg_type = "EA"

        def __init__(self):
            super().__init__(verboseFlag=False, logFlag=False, saveFlag=False)
            self.calls = 0

        def run(self, problem, seed=None):
            assert problem.optType == "min"
            candidate = np.array([[np.log(lengths[self.calls])]])
            self.calls += 1
            return SimpleNamespace(bestDecs=candidate, bestObjs=problem.evaluate(candidate).objs)

    X = np.linspace(0, 1, 12).reshape(-1, 1)
    Y = np.sin(6 * X)
    optimizer = CandidateEA()
    kernel = RBF(length_attr={"lb": 0.1, "ub": 3.0, "type": "float", "log": True})
    model = GPR(kernel=kernel, C=1e-6, C_attr=None, optimizer=optimizer, nRestartTimes=2).fit(X, Y)
    covariance = referenceKernel(X, X, 0.36, "rbf") + 1e-6 * np.eye(len(X))
    assert optimizer.calls == 3
    assert float(np.asarray(model.setting.get("l")).item()) == pytest.approx(0.36)
    assert model.fitState["objective"] == pytest.approx(referenceObjective(covariance, Y), rel=1e-7)
    np.testing.assert_allclose(model.predict(X), referenceKernel(X, X, 0.36, "rbf") @ np.linalg.solve(covariance, Y), atol=1e-8)
