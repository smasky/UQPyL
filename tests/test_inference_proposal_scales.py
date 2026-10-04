"""独立验证提议尺度、输入单位换算与 AMH 自适应协方差。"""

import numpy as np
import pytest

from UQPyL.inference import AMH, MH, MH_Gibbs
from UQPyL.inference.chain import Chain
from UQPyL.problem import Problem


methodClasses = [MH, AMH, MH_Gibbs]
distributions = ["gauss", "uniform"]


def newAlgorithm(methodClass, distribution, nChains=3, warmUp=0, maxIters=12):
    return methodClass(
        nChains=nChains,
        propDist=distribution,
        warmUp=warmUp,
        maxIters=maxIters,
        saveFlag=False,
        logFlag=False,
        verboseFlag=False,
    )


def propose(algorithm, current, covariance, lower, upper, distribution, dim=0):
    if isinstance(algorithm, MH_Gibbs):
        return algorithm.f_prop(dim, current, distribution, covariance, upper, lower)
    return algorithm.f_prop(current, distribution, covariance, upper, lower)


@pytest.mark.parametrize("methodClass", methodClasses)
@pytest.mark.parametrize("distribution", distributions)
@pytest.mark.parametrize("nInput", [1, 3])
def testProposalMatchesIndependentRandomGeneratorAtDeclaredScale(methodClass, distribution, nInput):
    nChains, seed = 3, 641
    lower, upper = np.full((1, nInput), -10.0), np.full((1, nInput), 10.0)
    current = np.arange(nChains * nInput, dtype=float).reshape(nChains, nInput) / 10
    scales = np.array([np.arange(1, nInput + 1) * (0.1 + 0.02 * chain) for chain in range(nChains)])
    covariance = [np.diag(row**2) for row in scales]
    originals = [current.copy(), *[value.copy() for value in covariance]]
    algorithm = newAlgorithm(methodClass, distribution, nChains)
    algorithm.rng = np.random.default_rng(seed)
    referenceRng = np.random.default_rng(seed)
    expected = current.copy()
    dimension = nInput - 1
    for chain in range(nChains):
        if methodClass is MH_Gibbs:
            if distribution == "gauss":
                expected[chain, dimension] = referenceRng.normal(current[chain, dimension], scales[chain, dimension])
            else:
                expected[chain, dimension] += referenceRng.uniform(-scales[chain, dimension], scales[chain, dimension])
        elif distribution == "gauss":
            expected[chain] = referenceRng.multivariate_normal(current[chain], np.diag(scales[chain] ** 2))
        else:
            expected[chain] += referenceRng.uniform(-scales[chain], scales[chain])
    assert np.all((expected > lower) & (expected < upper))
    actual = propose(algorithm, current, covariance, lower, upper, distribution, dimension)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    assert algorithm.rng.random() == referenceRng.random()
    for value, original in zip([current, *covariance], originals):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize(
    "methodClass,distribution",
    [
        (MH_Gibbs, "gauss"),
        (MH, "uniform"),
        (AMH, "uniform"),
        (MH_Gibbs, "uniform"),
    ],
)
def testProposalVarianceMatchesDistributionDefinition(methodClass, distribution):
    count, scale = 4096, 0.1
    algorithm = newAlgorithm(methodClass, distribution, count)
    algorithm.rng = np.random.default_rng(643)
    current = np.zeros((count, 1))
    covariance = [np.array([[scale**2]]) for _ in range(count)]
    samples = propose(algorithm, current, covariance, np.array([[-10.0]]), np.array([[10.0]]), distribution)
    expectedVariance = scale**2 if distribution == "gauss" else scale**2 / 3
    assert np.mean(samples) == pytest.approx(0.0, abs=scale * 0.1)
    assert np.var(samples) == pytest.approx(expectedVariance, rel=0.06)
    if distribution == "uniform":
        assert np.max(np.abs(samples)) <= scale


@pytest.mark.parametrize("methodClass", methodClasses)
@pytest.mark.parametrize("distribution", distributions)
@pytest.mark.parametrize("warmUp", [0, 3])
def testPublicSamplingTraceIsInvariantToInputUnitsThroughAdaptiveUpdates(methodClass, distribution, warmUp):
    normalized = []
    for span in [0.1, 1.0, 10.0]:
        count = [0]

        def objective(x):
            count[0] += len(x)
            return np.zeros((len(x), 1))

        problem = Problem(nInput=1, nObj=1, lb=0, ub=span, objFunc=objective)
        result = newAlgorithm(methodClass, distribution, nChains=4, warmUp=warmUp).run(problem, gamma=0.1, seed=641)
        assert result.decs.shape == (4, 12, 1)
        assert result.FEs == count[0]
        if methodClass is AMH:
            # Rejected out-of-box proposals do not call the objective.
            assert 4 <= result.FEs <= 4 * (warmUp + 12)
            rejected = ~result.accepted[:, 1:]
            np.testing.assert_array_equal(result.decs[:, 1:][rejected], result.decs[:, :-1][rejected])
        else:
            assert result.FEs == 4 * (warmUp + 12)
        assert result.iters == 11
        assert np.all(result.feasibleMask)
        if methodClass is not AMH:
            assert np.all(result.accepted)
        assert np.all((result.decs >= 0) & (result.decs <= span))
        np.testing.assert_array_equal(result.objs, np.zeros((4, 12, 1)))
        np.testing.assert_array_equal(result.logProb, np.zeros((4, 12)))
        normalized.append(result.decs / span)
    for values in normalized[1:]:
        np.testing.assert_allclose(values, normalized[0], rtol=0, atol=3e-12)


@pytest.mark.parametrize("unitScale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("constantHistory", [False, True])
def testAdaptiveCovarianceAndFloorUseParameterUnits(unitScale, constantHistory):
    baseSpan = np.array([1.0, 3.0])
    span = baseSpan * unitScale
    unitHistory = np.array([[0.2, 0.4], [0.25, 0.5], [0.4, 0.7]])
    if constantHistory:
        unitHistory[:] = [0.25, 0.5]
    history = unitHistory * span
    chain = Chain(2, 1, 0, 3)
    for values in history:
        chain.add(values, np.zeros(1))
    algorithm = newAlgorithm(AMH, "gauss", 1)
    algorithm.setProblem(Problem(nInput=2, nObj=1, lb=0, ub=span, objFunc=lambda x: np.zeros((len(x), 1))))
    multiplier = 2.38**2 / 2
    centered = history - history.mean(axis=0)
    expected = sum(np.outer(row, row) for row in centered) / 2 * multiplier + np.diag(1e-3 * span**2)
    actual = algorithm.updateCovs([chain], multiplier)[0]
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    assert np.all(np.linalg.eigvalsh(actual) > 0)


def testAdaptiveFallbackRespectsBoundsAndRetainsProvidedCovarianceCopy():
    algorithm = newAlgorithm(AMH, "gauss", 1)
    problem = Problem(nInput=2, nObj=1, lb=0, ub=[1.0, 3.0], objFunc=lambda x: np.zeros((len(x), 1)))
    algorithm.setProblem(problem)
    chain = Chain(2, 1, 0, 3)
    chain.add(np.array([0.5, 1.5]), np.zeros(1))
    np.testing.assert_allclose(algorithm.updateCovs([chain], 2.0)[0], np.diag([2.0, 18.0]))
    original = np.array([[0.04, 0.02], [0.02, 0.09]])
    restored = algorithm.updateCovs([chain], 2.0, [original])[0]
    np.testing.assert_array_equal(restored, original)
    restored[0, 0] = 10
    assert original[0, 0] == 0.04


@pytest.mark.parametrize("methodClass", methodClasses)
@pytest.mark.parametrize("distribution", distributions)
def testFixedParameterStaysFixedAfterWarmUpAndSampling(methodClass, distribution):
    problem = Problem(nInput=2, nObj=1, lb=[0.0, 2.0], ub=[1.0, 2.0], objFunc=lambda x: np.zeros((len(x), 1)))
    algorithm = newAlgorithm(methodClass, distribution, nChains=2, warmUp=3)
    result = algorithm.run(problem, gamma=0.1, seed=647)
    np.testing.assert_array_equal(result.decs[:, :, 1], np.full((2, 12), 2.0))
    assert np.all((result.decs[:, :, 0] >= 0) & (result.decs[:, :, 0] <= 1))


@pytest.mark.parametrize("methodClass", [MH, AMH])
def testGaussianProposalStillUsesFullCorrelatedCovariance(methodClass):
    covariance = [np.array([[0.04, 0.02], [0.02, 0.09]])] * 3
    current = np.zeros((3, 2))
    algorithm = newAlgorithm(methodClass, "gauss")
    algorithm.rng = np.random.default_rng(651)
    referenceRng = np.random.default_rng(651)
    expected = np.array([referenceRng.multivariate_normal(row, covariance[index]) for index, row in enumerate(current)])
    actual = propose(algorithm, current, covariance, np.full((1, 2), -10.0), np.full((1, 2), 10.0), "gauss")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
