"""Independent transition identities and known-target distribution regressions."""

import numpy as np
import pytest

from UQPyL.inference import AMH, DEMC, DREAM_ZS
from UQPyL.problem import Problem


quiet = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def flatProblem(nInput=2, lb=-10.0, ub=10.0):
    return Problem(nInput=nInput, nObj=1, lb=lb, ub=ub, objFunc=lambda x: np.zeros((len(x), 1)))


@pytest.mark.parametrize("methodClass", [AMH, DEMC, DREAM_ZS])
def testOutsideProposalsNeverReachModelOrCountAsAccepted(methodClass):
    class Outside(methodClass):
        def f_prop(self, current, *args, **kwargs):
            shape = (1, current.shape[1]) if kwargs.get("chainIndex") is not None else current.shape
            return np.full(shape, 100.0)

        def f_prop_ratio(self, current, *args, **kwargs):
            return np.full_like(current, 100.0), np.zeros(len(current)), np.zeros(len(current), dtype=int)

    seen = []

    def objective(x):
        assert np.all(np.abs(x) <= 1.0)
        seen.extend(x.copy())
        return np.sum(x * x, axis=1, keepdims=True)

    problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objective)
    result = Outside(nChains=4, warmUp=3, maxIters=6, **quiet).run(problem, seed=17)
    assert result.FEs == len(seen) == 4
    assert not result.accepted[:, 1:].any()
    np.testing.assert_array_equal(result.decs, np.repeat(result.decs[:, :1], 6, axis=1))


def testAmhRetainsRawCorrelatedProposalForRejection():
    method = AMH(**quiet)

    class ProposalRng:
        def multivariate_normal(self, *args):
            return np.array([1.4, -0.2])

    method.rng = ProposalRng()
    raw = method.f_prop(np.array([[0.5, 0.5]]), "gauss", [np.eye(2)], np.ones((1, 2)), np.zeros((1, 2)))
    np.testing.assert_array_equal(raw, [[1.4, -0.2]])


@pytest.mark.parametrize("warmUp", [0, 2])
def testDemcConditionsEachChainOnEarlierAcceptedUpdates(warmUp):
    class Probe(DEMC):
        def initialSampling(self, problem, nChains):
            current = np.arange(nChains, dtype=float)[:, None]
            objs, cons = self.evaluate(current)
            self.expected = current.copy()
            self.calls = []
            return current, objs, cons

        def f_prop(self, current, ub, lb, gamma=None, chainIndex=None):
            assert chainIndex is not None
            np.testing.assert_array_equal(current, self.expected)
            proposed = current[chainIndex : chainIndex + 1] + 0.1
            self.expected[chainIndex] = proposed[0]
            self.calls.append(chainIndex)
            return proposed

    method = Probe(nChains=3, warmUp=warmUp, maxIters=4, **quiet)
    result = method.run(flatProblem(nInput=1), seed=5)
    assert method.calls == [0, 1, 2] * (warmUp + 3)
    np.testing.assert_allclose(result.decs[:, -1, 0], np.arange(3) + 0.1 * (warmUp + 3))


@pytest.mark.numerical
@pytest.mark.statistical
def testDemcNoiseIsCenteredIndependentAndEscapesCollapsedPopulation():
    method = DEMC(nChains=3, **quiet)
    method.setup(flatProblem(lb=-1.0, ub=1.0), seed=17)
    current = np.zeros((3, 2))
    samples = np.concatenate([method.f_prop(current, method.problem.ub, method.problem.lb) for _ in range(1500)])
    normalized = samples / 2e-6
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(np.cov(normalized.T), np.eye(2), atol=0.06)


class FixedDonors:
    def __init__(self, reverse=False):
        self.reverse = reverse

    def choice(self, *args, **kwargs):
        return np.array([0, 2, 1] if self.reverse else [0, 1, 2])


@pytest.mark.parametrize("units", [np.ones(2), np.array([0.01, 100.0])])
@pytest.mark.numerical
def testSnookerProjectionAndReverseHastingsRatio(units):
    method = DREAM_ZS(**quiet)
    method.setup(flatProblem(lb=-10 * units, ub=10 * units), seed=5)
    method.rng = FixedDonors()
    archive = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, 0.5]]) * units
    current = np.zeros((3, 2))
    proposed, ratio = method.snooker_update(0, current, archive, 1.7)
    np.testing.assert_allclose(proposed / units, [1.36, 2.72], atol=1e-14)
    assert ratio == pytest.approx(0.36)
    method.rng = FixedDonors(reverse=True)
    current[0] = proposed
    reverse, reverseRatio = method.snooker_update(0, current, archive, 1.7)
    np.testing.assert_allclose(reverse / units, [0.0, 0.0], atol=1e-14)
    assert ratio * reverseRatio == pytest.approx(1.0)


@pytest.mark.parametrize("fixed", [False, True])
def testOneActiveDimensionHasUnitSnookerCorrection(fixed):
    lower, upper = ([-10.0, 7.0], [10.0, 7.0]) if fixed else ([-10.0], [10.0])
    method = DREAM_ZS(**quiet)
    method.setup(flatProblem(nInput=len(lower), lb=lower, ub=upper), seed=5)
    method.rng = FixedDonors()
    current = np.array([[0.0, 7.0]] * 3) if fixed else np.zeros((3, 1))
    archive = np.array([[0.5, 7.0], [2.0, 7.0], [-1.0, 7.0]]) if fixed else np.array([[0.5], [2.0], [-1.0]])
    proposed, ratio = method.snooker_update(0, current, archive, 1.7)
    assert ratio == 1.0
    if fixed:
        assert proposed[1] == 7.0


def testZeroSnookerAxisIsFiniteSelfTransition():
    method = DREAM_ZS(**quiet)
    method.setup(flatProblem(), seed=5)
    method.rng = FixedDonors()
    proposed, correction = method.snooker_update(0, np.zeros((3, 2)), np.zeros((3, 2)), 1.7, logRatio=True)
    np.testing.assert_array_equal(proposed, [0.0, 0.0])
    assert correction == 0.0


def testSnookerLogCorrectionSurvivesHugeDensityRatios():
    dimension = 100
    method = DREAM_ZS(**quiet)
    method.setup(flatProblem(nInput=dimension, lb=-1.0, ub=1.0), seed=5)
    method.rng = FixedDonors()
    current = np.full((3, dimension), 1e-10)
    archive = np.array([np.zeros(dimension), np.full(dimension, 0.1), np.full(dimension, -0.1)])
    proposed, correction = method.snooker_update(0, current, archive, 1.0, logRatio=True)
    assert correction > 1000 and np.isfinite(correction)
    method.rng = np.random.default_rng(17)
    # An unfavorable target ratio must offset the large proposal correction.
    assert not method.accept([correction + 1000], [0.0], decStar=proposed, decCur=current[0], logQRatio=correction)


def testDreamAdaptationUsesActualWarmupWindowAndSquaredJumpNorm():
    class Probe(DREAM_ZS):
        def initialSampling(self, problem, nChains):
            self.calls, self.reports = 0, []
            current = np.zeros((nChains, 2))
            objs, cons = self.evaluate(current)
            return current, objs, cons

        def f_prop_ratio(self, current, archive, *args, **kwargs):
            args[7][0] += len(current)  # cr_tries, independent forced proposal.
            return current + [0.01, -0.01], np.zeros(len(current)), np.zeros(len(current), dtype=int)

        def accept(self, *args, **kwargs):
            accepted = (self.calls // 3) % 2 == 0
            self.calls += 1
            return accepted

        def adaption(self, pCR, gains, tries, rates, scale, target):
            self.reports.append((gains.copy(), rates.copy()))
            return super().adaption(pCR, gains, tries, rates, scale, target)

    method = Probe(nChains=3, warmUp=7, maxIters=9, adpInterval=4, **quiet)
    result = method.run(flatProblem(), seed=5)
    assert len(method.reports) == 2  # No adaptation during formal sampling.
    np.testing.assert_allclose(method.reports[0][1], 0.5)
    np.testing.assert_allclose(method.reports[1][1], 2 / 3)
    assert all(gains[0] > 0 for gains, _ in method.reports)
    assert result.diagnostics["sampler"]["gamma_scale"] == pytest.approx(np.exp(0.1 * (0.5 + 2 / 3 - 0.5)))


@pytest.mark.parametrize("methodClass", [DEMC, DREAM_ZS])
def testDifferentialSamplingIsInvariantToParameterUnits(methodClass):
    traces = []
    for span, offset in [(np.ones(2), np.zeros(2)), (np.array([0.001, 1000.0]), np.array([1.0, -2.0]))]:

        def objective(x):
            unit = (x - offset) / span
            return 10 * np.sum((unit - 0.5) ** 2, axis=1, keepdims=True)

        problem = Problem(nInput=2, nObj=1, lb=offset, ub=offset + span, objFunc=objective)
        result = methodClass(nChains=4, warmUp=12, maxIters=40, **quiet).run(problem, seed=17)
        traces.append((result.decs - offset) / span)
    np.testing.assert_allclose(traces[0], traces[1], atol=1e-8, rtol=0)


@pytest.mark.parametrize("methodClass", [AMH, DEMC, DREAM_ZS])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.numerical
@pytest.mark.statistical
def testKnownGaussianMoments(methodClass, bounded):
    covariance = np.array([[1.0, 0.9], [0.9, 1.0]]) if bounded else np.eye(2)
    precision = np.linalg.inv(covariance)
    problem = Problem(
        nInput=2,
        nObj=1,
        lb=-1.0 if bounded else -6.0,
        ub=1.0 if bounded else 6.0,
        objFunc=lambda x: 0.5 * np.einsum("ni,ij,nj->n", x, precision, x)[:, None],
    )
    result = methodClass(nChains=4, warmUp=1000, maxIters=4000, **quiet).run(problem, seed=17)
    samples = result.decs[:, 1000:].reshape(-1, 2)
    if bounded:
        # Independent Gauss-Legendre quadrature of the bounded target.
        nodes, weights = np.polynomial.legendre.leggauss(60)
        grid = np.stack(np.meshgrid(nodes, nodes), axis=-1).reshape(-1, 2)
        mass = np.outer(weights, weights).ravel() * np.exp(-0.5 * np.einsum("ni,ij,nj->n", grid, precision, grid))
        mass /= mass.sum()
        covariance = np.einsum("n,ni,nj->ij", mass, grid, grid)
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.1)
    np.testing.assert_allclose(np.cov(samples.T), covariance, atol=0.027 if bounded else 0.12)


@pytest.mark.parametrize(
    "key,value",
    [
        ("k", 0),
        ("k", 51),
        ("archSize", 0),
        ("nCR", 0),
        ("adpInterval", 0),
        ("warmUp", -1),
        ("ps", 1.1),
        ("jitter", -1.0),
        ("acTarget", np.nan),
    ],
)
def testDreamRejectsUndefinedProposalConfiguration(key, value):
    with pytest.raises(ValueError):
        DREAM_ZS(**{key: value}, **quiet).run(flatProblem(), seed=17)
