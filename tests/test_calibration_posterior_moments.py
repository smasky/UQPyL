"""Analytic posterior and prior-anchored RML controls, independent of gain code."""

import numpy as np
import pytest

from UQPyL.calibration import ES, IES
from UQPyL.problem import ModelProblem


def linearProblem(matrix, obs, bounds=1e6):
    return ModelProblem(
        nInput=matrix.shape[1],
        lb=-bounds,
        ub=bounds,
        obs=(np.asarray(obs)[:, None]).reshape(-1),
        simFunc=lambda x: x @ matrix.T,
    )


def targetsFor(obs, covariance, count, seed):
    values, vectors = np.linalg.eigh(covariance)
    return obs + np.random.default_rng(seed).standard_normal((count, len(obs))) @ (vectors * np.sqrt(values)).T


@pytest.mark.parametrize("dimension,count", [(1, 3), (2, 20), (8, 40), (20, 12)])
@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.numerical
def testSquareRootMatchesAnalyticKalmanMoments(dimension, count, seed):
    rng = np.random.default_rng(seed)
    prior = rng.normal(size=(count, dimension))
    matrix = rng.normal(size=(2, dimension))
    noise = np.array([[0.5, 0.1], [0.1, 0.8]])
    obs = np.array([0.6, -0.2])
    covariance = np.atleast_2d(np.cov(prior, rowvar=False))
    gain = np.linalg.solve(matrix @ covariance @ matrix.T + noise, matrix @ covariance).T
    expectedMean = prior.mean(0) + gain @ (obs - matrix @ prior.mean(0))
    expectedCovariance = covariance - gain @ matrix @ covariance
    saved = prior.copy()
    result = ES().run(linearProblem(matrix, obs), prior, r=noise)
    np.testing.assert_allclose(result.posteriorDecs.mean(0), expectedMean, atol=2e-12)
    np.testing.assert_allclose(
        np.atleast_2d(np.cov(result.posteriorDecs, rowvar=False)), expectedCovariance, atol=2e-12
    )
    np.testing.assert_array_equal(prior, saved)


@pytest.mark.numerical
def testOriginalScalarUnderdispersionIsFixed():
    result = ES().run(linearProblem(np.ones((1, 1)), [1]), [[-1], [0], [1]], r=np.ones((1, 1)))
    assert result.posteriorDecs.mean() == pytest.approx(0.5)
    assert result.posteriorDecs.var(ddof=1) == pytest.approx(0.5)


@pytest.mark.parametrize("count,dimension", [(24, 2), (12, 20)])
@pytest.mark.parametrize("iterations", [1, 3, 10])
@pytest.mark.numerical
def testIesLinearMembersMatchFixedRandomizedMap(count, dimension, iterations):
    rng = np.random.default_rng(9)
    prior = rng.normal(size=(count, dimension))
    matrix = rng.normal(size=(2, dimension))
    noise = np.array([[0.5, 0.1], [0.1, 0.8]])
    obs = np.array([0.6, -0.2])
    covariance = np.atleast_2d(np.cov(prior, rowvar=False))
    gain = np.linalg.solve(matrix @ covariance @ matrix.T + noise, matrix @ covariance).T
    targets = targetsFor(obs, noise, count, 12)
    expected = prior + (targets - prior @ matrix.T) @ gain.T
    result = IES(maxIters=iterations, seed=12).run(linearProblem(matrix, obs), prior, r=noise)
    np.testing.assert_allclose(result.posteriorDecs, expected, atol=5e-11)


@pytest.mark.parametrize("lam", [0, 0.5, 3])
@pytest.mark.numerical
def testDampingMatchesIndependentNormalEquations(lam):
    rng = np.random.default_rng(51)
    prior = rng.normal(size=(30, 2))
    matrix = np.array([[1.0, 2.0], [-1.0, 3.0]])
    noise = np.array([[0.5, 0.1], [0.1, 0.8]])
    obs = np.array([0.3, -0.4])
    precision = np.linalg.inv(np.cov(prior, rowvar=False))
    noisePrecision = np.linalg.inv(noise)
    targets = targetsFor(obs, noise, len(prior), 15)
    expected = prior.copy()
    hessian = (1 + lam) * precision + matrix.T @ noisePrecision @ matrix
    for _ in range(4):
        gradient = (expected - prior) @ precision + (expected @ matrix.T - targets) @ noisePrecision @ matrix
        expected -= np.linalg.solve(hessian, gradient.T).T
    actual = IES(maxIters=4, lam=lam, seed=15).run(linearProblem(matrix, obs), prior, r=noise)
    np.testing.assert_allclose(actual.posteriorDecs, expected, atol=2e-12)


@pytest.mark.parametrize("noise", [None, np.zeros((2, 2)), np.diag([0.0, 1.0])])
@pytest.mark.parametrize("methodClass", [ES, IES])
@pytest.mark.numerical
def testHardObservationAndUnobservedDirection(noise, methodClass):
    prior = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    # Second observation is constant, so cannot inform the second parameter.
    matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
    options = dict(maxIters=6, seed=5) if methodClass is IES else {}
    result = methodClass(**options).run(linearProblem(matrix, [0.5, 0.0]), prior, r=noise)
    np.testing.assert_allclose(result.posteriorDecs[:, 0], 0.5, atol=2e-8)
    np.testing.assert_allclose(result.posteriorDecs[:, 1], prior[:, 1], atol=2e-12)


def testIesSeedReproducibilityAndGlobalRngIsolation():
    prior = np.linspace(-2, 2, 30)[:, None]
    problem = linearProblem(np.ones((1, 1)), [1])
    method = IES(seed=42)
    np.random.seed(975)
    before = np.random.get_state()
    first = method.run(problem, prior, r=np.ones((1, 1)))
    second = method.run(problem, prior, r=np.ones((1, 1)))
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(first.posteriorDecs, second.posteriorDecs)
    other = IES(seed=43).run(problem, prior, r=np.ones((1, 1)))
    assert not np.array_equal(first.posteriorDecs, other.posteriorDecs)


@pytest.mark.parametrize("methodClass", [ES, IES])
def testNoisyProjectionAndMaskAreAppliedBeforeSimulation(methodClass):
    calls = []

    def simulate(x):
        assert np.all((x >= -1) & (x <= 1))
        calls.append(x.copy())
        return np.column_stack((x[:, 0], x[:, 0]))

    problem = ModelProblem(
        nInput=1, lb=-1, ub=1, obs=np.array([100.0, np.nan]), mask=np.array([False, True]), simFunc=simulate
    )
    options = dict(maxIters=3, seed=6) if methodClass is IES else {}
    result = methodClass(**options).run(problem, [[-0.5], [0], [0.5]], r=np.array([[0.1]]))
    assert len(calls) == (4 if methodClass is IES else 2)
    assert result.diagnostics["boundUpdates"][0]["adjusted_members"] == 3
    np.testing.assert_array_equal(result.posteriorDecs, np.ones((3, 1)))


@pytest.mark.numerical
@pytest.mark.statistical
def testStochasticIesRecoversScalarPosteriorMomentsAtLargeSample():
    prior = np.random.default_rng(4).normal(size=(20000, 1))
    prior = (prior - prior.mean()) / prior.std(ddof=1)
    result = IES(maxIters=3, seed=43).run(linearProblem(np.ones((1, 1)), [1]), prior, r=np.ones((1, 1)))
    assert abs(result.posteriorDecs.mean() - 0.5) < 0.015
    assert abs(result.posteriorDecs.var(ddof=1) - 0.5) < 0.015


@pytest.mark.parametrize("factor", [1e-6, 1e6])
def testDampingDoesNotDependOnObservationUnits(factor):
    prior = np.linspace(-2, 2, 30)[:, None]
    standard = IES(maxIters=4, lam=2.0, seed=19).run(linearProblem(np.ones((1, 1)), [1]), prior, r=np.ones((1, 1)))
    scaled = IES(maxIters=4, lam=2.0, seed=19).run(
        linearProblem(np.ones((1, 1)) * factor, [factor]), prior, r=np.ones((1, 1)) * factor**2
    )
    np.testing.assert_allclose(scaled.posteriorDecs, standard.posteriorDecs, atol=2e-12)


def testIesRegressionIsStableAcrossParameterUnits():
    prior = np.random.default_rng(31).normal(size=(24, 2))
    matrix = np.array([[1.0, 2.0], [-1.0, 0.5]])
    units = np.array([1e-8, 1e8])
    options = dict(maxIters=4, lam=0.5, seed=15)
    ordinary = IES(**options).run(linearProblem(matrix, [0.5, 1.0]), prior, r=np.eye(2))
    scaled = IES(**options).run(linearProblem(matrix / units, [0.5, 1.0], bounds=1e12), prior * units, r=np.eye(2))
    np.testing.assert_allclose(scaled.posteriorDecs / units, ordinary.posteriorDecs, atol=2e-12)
