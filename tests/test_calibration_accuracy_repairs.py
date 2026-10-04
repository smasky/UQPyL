"""Accuracy controls with analytic posteriors and independent nonlinear MAP."""

import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.stats import norm, truncnorm

from UQPyL.calibration import IES, SUFI2, CalReader
from UQPyL.problem import ModelProblem


def scalarProblem(observation=1.0, lb=-6.0, ub=6.0):
    return ModelProblem(
        nInput=1,
        lb=lb,
        ub=ub,
        obs=np.array([observation, 2 * observation]),
        simFunc=lambda x: np.stack([x[:, 0], 2 * x[:, 0]], axis=1),
    )


@pytest.mark.numerical
def testWeightedUncertaintyMatchesGaussianPosteriorAndSqlite(tmp_path):
    prior = norm.ppf((np.arange(4096) + 0.5) / 4096)[:, None]
    problem = scalarProblem()
    problem.workDir = str(tmp_path)
    result = SUFI2(saveFlag=True).run(
        problem,
        [[-1.0], [0.0], [1.0]],
        eliteSize=1,
        uncertaintyX=prior,
        logLikelihood=lambda obs, sim, mask: -0.5 * ((sim[:, 0] - obs[0]) / 0.5) ** 2,
    )
    u = result.extra["uncertainty"]
    np.testing.assert_allclose(u["parameter_mean"], [0.8], atol=5e-4)
    np.testing.assert_allclose(u["parameter_variance"], [0.2], atol=5e-4)
    np.testing.assert_allclose(
        [u["parameter_lower"][0], u["parameter_upper"][0]],
        norm.ppf([0.025, 0.975], loc=0.8, scale=np.sqrt(0.2)),
        atol=0.005,
    )
    assert result.diagnostics["updatedLb"][0] == result.diagnostics["updatedUb"][0] == 1.0
    with CalReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        restored = reader.load_result()
    np.testing.assert_array_equal(restored.extra["uncertainty"]["weights"], u["weights"])
    np.testing.assert_array_equal(restored.extra["uncertainty"]["parameter_lower"], u["parameter_lower"])


@pytest.mark.numerical
def testGeneratedUncertaintyUsesOriginalUniformDomainNotEliteBounds():
    result = SUFI2(nSamples=24, maxIters=4).run(
        scalarProblem(0.7, 0, 1),
        eliteSize=1,
        seed=11,
        uncertaintySamples=4096,
        logLikelihood=lambda obs, sim, mask: -0.5 * ((sim[:, 0] - 0.7) / 0.1) ** 2,
    )
    u = result.extra["uncertainty"]
    exact = truncnorm(-7, 3, loc=0.7, scale=0.1)
    assert u["prior_source"] == "original_domain_uniform"
    assert u["samples"].min() < 0.001 and u["samples"].max() > 0.999
    np.testing.assert_allclose(u["parameter_mean"], exact.mean(), atol=2e-4)
    np.testing.assert_allclose(u["parameter_variance"], exact.var(), atol=2e-5)


@pytest.mark.parametrize("seed", [3, 11, 23])
def testIndependentUncertaintyDoesNotChangeSearchAndRetainsIncumbent(seed):
    problem = scalarProblem(0.731, 0, 1)
    options = dict(nSamples=12, maxIters=5)
    ordinary = SUFI2(**options).run(problem, eliteSize=1, seed=seed)
    weighted = SUFI2(**options).run(
        problem, eliteSize=1, seed=seed, logLikelihood=lambda obs, sim, mask: -0.5 * ((sim[:, 0] - 0.731) / 0.2) ** 2
    )
    np.testing.assert_array_equal(ordinary.posteriorDecs, weighted.posteriorDecs)
    np.testing.assert_array_equal(ordinary.bestDecs, weighted.bestDecs)
    scores = [item["bestScore"] for item in ordinary.history.metricsHistory]
    assert np.all(np.diff(scores) <= 0)
    assert ordinary.diagnostics["uncertaintyStatus"] == "not_estimated"
    assert ordinary.diagnostics["intervalKind"] == "sampling_envelope"


def testLowEffectiveSampleSizeWarnsAndPreservesWeights():
    with pytest.warns(RuntimeWarning, match="effective sample size"):
        result = SUFI2().run(
            scalarProblem(),
            [[0.0], [1.0]],
            eliteSize=1,
            uncertaintyX=[[0.0], [1.0]],
            logLikelihood=lambda obs, sim, mask: np.array([-np.inf, 0.0]),
        )
    u = result.extra["uncertainty"]
    assert u["effective_sample_size"] == 1
    assert result.diagnostics["uncertaintyStatus"] == "low_effective_sample_size"
    np.testing.assert_array_equal(u["parameter_lower"], [1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(uncertaintyX=[[0], [1]]),
        dict(logLikelihood=3),
        dict(interval=1),
        dict(logLikelihood=lambda *a, **k: None, uncertaintySamples=1),
        dict(logLikelihood=lambda *a, **k: None, uncertaintyX=[[999], [0]]),
    ],
)
def testUncertaintyPreflightBeforeSimulation(kwargs):
    problem = scalarProblem()
    problem.simFunc = lambda x: pytest.fail("Invalid inputs must not simulate")
    with pytest.raises(ValueError):
        SUFI2().run(problem, [[0.0], [1.0]], eliteSize=1, **kwargs)


@pytest.mark.parametrize("seed", [1, 5, 12])
@pytest.mark.numerical
def testLocalRmlMatchesIndependentScalarMap(seed):
    prior = np.random.default_rng(seed).normal(size=(24, 1))
    problem = ModelProblem(
        nInput=1, lb=-10, ub=10, obs=np.array([1.0]), simFunc=lambda x: x + 0.3 * x ** 3
    )
    sigma = 0.3
    targets = 1 + np.random.default_rng(seed + 100).standard_normal(prior.shape) * sigma
    priorStd = prior.std(ddof=1)
    expected = []
    for initial, target in zip(prior[:, 0], targets[:, 0]):
        solve = least_squares(
            lambda z: np.array([(z[0] - initial) / priorStd, (z[0] + 0.3 * z[0] ** 3 - target) / sigma]),
            x0=[initial],
            bounds=(-10, 10),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
        )
        expected.append(solve.x[0])
    result = IES(maxIters=50, seed=seed + 100, adaptive=True, localLinearization=True, tolerance=1e-8).run(
        problem, prior, r=np.array([[sigma**2]])
    )
    np.testing.assert_allclose(result.posteriorDecs[:, 0], expected, atol=2e-5)
    assert result.diagnostics["stopReason"] == "step_tolerance"


@pytest.mark.parametrize("local", [False, True])
def testLocalModePreservesLinearSolution(local):
    rng = np.random.default_rng(15)
    prior = rng.normal(size=(32, 2))
    matrix = np.array([[1.0, 2.0], [-1.0, 0.5]])
    problem = ModelProblem(
        nInput=2, lb=-100, ub=100, obs=np.array([0.5, 1.0]), simFunc=lambda x: x @ matrix.T
    )
    expected = IES(maxIters=1, seed=9).run(problem, prior, r=np.eye(2))
    actual = IES(maxIters=3, seed=9, localLinearization=local).run(problem, prior, r=np.eye(2))
    np.testing.assert_allclose(actual.posteriorDecs, expected.posteriorDecs, atol=1e-8)


def testLocalPerturbationsRespectBoundsFixedParametersAndMask():
    calls = []

    def simulate(x):
        assert np.all((x[:, 0] >= 0) & (x[:, 0] <= 1))
        assert np.all(x[:, 1] == 2.0)
        calls.append(x.copy())
        return np.column_stack([x[:, 0] ** 2, x[:, 0], 100 * x[:, 0]])

    problem = ModelProblem(
        nInput=2,
        lb=[0, 2],
        ub=[1, 2],
        obs=np.array([0.5, 0.7, 0.0]),
        mask=np.array([False, False, True]),
        simFunc=simulate,
    )
    result = IES(maxIters=1, seed=2, localLinearization=True).run(
        problem, [[0.0, 2.0], [0.5, 2.0], [1.0, 2.0]], r=np.eye(2) * 0.1
    )
    assert len(calls) == 4  # Initial, plus/minus one active parameter, posterior.
    assert np.isfinite(result.posteriorDecs).all()
    assert result.diagnostics["linearization"] == "member_finite_difference"
