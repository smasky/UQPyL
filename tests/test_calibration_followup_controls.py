"""Independent controls for sampling guards, weighted intervals and safe steps."""

import numpy as np
import pytest

from UQPyL.calibration import ES, IES, GLUE, SUFI2, CalReader
from UQPyL.problem import ModelProblem


def simpleProblem(**kwargs):
    return ModelProblem(nInput=1, lb=0., ub=1., obs=np.array([0.731, 1.462]),
                        simFunc=lambda x: np.stack([x[:, 0], 2 * x[:, 0]], axis=1), **kwargs)


@pytest.mark.parametrize('value', [-1, 1.5, True])
def testSufiIterationValidationBeforeSimulation(value):
    problem = simpleProblem()
    problem.simFunc = lambda x: pytest.fail('Invalid configuration must not simulate')
    with pytest.raises(ValueError, match='maxIters'):
        SUFI2(maxIters=value).run(problem, [[.2], [.8]], eliteSize=1)


def testSufiZeroIterationsWarnsAndScreensOnce():
    with pytest.warns(RuntimeWarning, match='one screening'):
        result = SUFI2(maxIters=0).run(simpleProblem(), [[.2], [.8]], eliteSize=1)
    assert len(result.history.metricsHistory) == 1
    np.testing.assert_array_equal(result.bestDecs, [[.8]])


@pytest.mark.parametrize('options', [dict(nSamples=1.5), dict(nSamples=True), dict(explorationFraction=-.1),
                                    dict(minRangeFraction=np.nan)])
def testSufiRejectsInvalidSamplingOptions(options):
    with pytest.raises(ValueError):
        SUFI2(**options).run(simpleProblem(), eliteSize=1)


@pytest.mark.parametrize('seed', [11, 23, 47])
def testSingleEliteDoesNotFreezeSampling(seed):
    original = SUFI2(nSamples=12, maxIters=5, explorationFraction=0, minRangeFraction=0).run(
        simpleProblem(), eliteSize=1, seed=seed)
    guarded = SUFI2(nSamples=12, maxIters=5).run(simpleProblem(), eliteSize=1, seed=seed)
    assert np.ptp(original.posteriorDecs) == 0
    assert np.ptp(guarded.posteriorDecs) > .05
    for history in guarded.history.metricsHistory[1:]:
        assert history['samplingUb'][0] - history['samplingLb'][0] >= .05 - 1e-14
        assert history['explorationCount'] == 2
    assert guarded.diagnostics['scores'].min() < original.diagnostics['scores'].min()


@pytest.mark.parametrize('seed', [11, 23, 47])
def testGuardedMixedSamplingRetainsGlobalLegalSupport(seed):
    batches = []
    choices = [-2.25, .125, 3.5, 7.5]
    def simulate(x):
        batches.append(x.copy())
        assert np.isin(x[:, 1], [0, 1, 2]).all()
        assert np.isin(x[:, 2], choices).all()
        return np.column_stack([x.sum(1), 2 * x.sum(1)])
    problem = ModelProblem(nInput=3, lb=[0, 0, 0], ub=[1, 2, 1], varType=[0, 1, 2],
                           varSet={2: choices}, obs=np.array([1.0, 2.0]), simFunc=simulate)
    result = SUFI2(nSamples=24, maxIters=4).run(problem, eliteSize=1, seed=seed)
    for batch, history in zip(batches[1:], result.history.metricsHistory[1:]):
        assert history['explorationCount'] == 3
        assert np.ptp(batch[:, 0]) > 0
        assert len(np.unique(batch[-3:, 2])) >= 2


def testGlueWeightsAndDiscreteCdfQuantilesWithMask(tmp_path):
    problem = ModelProblem(nInput=1, lb=0, ub=10, obs=np.array([1.0, 999.0]),
                           mask=np.array([False, True]), simFunc=lambda x: (np.repeat(x[:, None, :], 2, axis=1)).reshape(len(x), -1))
    problem.workDir = str(tmp_path)
    def likelihood(obs, sims, mask):
        np.testing.assert_array_equal(mask, [False, True])
        return np.log([.1, .7, .2]) - 10000
    with pytest.warns(RuntimeWarning, match="effective sample size"):
        result = GLUE(saveFlag=True).run(problem, [[0.], [1.], [3.], [10.]], threshold=3.,
                                        logLikelihood=likelihood, interval=.5)
    np.testing.assert_allclose(result.diagnostics['behavioralWeights'], [.1, .7, .2], atol=1e-12)
    # The inverse empirical CDF at q=.25 and q=.75 is exactly 1.
    assert result.diagnostics['ppuLower'][0] == 1.
    assert result.diagnostics['ppuUpper'][0] == 1.
    assert result.diagnostics['effectiveSampleSize'] == pytest.approx(1/.54)
    with CalReader(next(tmp_path.rglob('*.sqlite3'))) as reader:
        restored = reader.load_result()
    np.testing.assert_array_equal(restored.diagnostics['behavioralWeights'], result.diagnostics['behavioralWeights'])


def testGlueUniformAndZeroLikelihoodSamples():
    problem = simpleProblem()
    uniform = GLUE().run(problem, [[0.], [.5], [1.]], threshold=10.)
    np.testing.assert_allclose(uniform.diagnostics['behavioralWeights'], [1/3]*3)
    with pytest.warns(RuntimeWarning, match="effective sample size"):
        weighted = GLUE().run(problem, [[0.], [.5], [1.]], threshold=10.,
                              logLikelihood=lambda obs, sims, mask: np.array([-np.inf, 0., -np.inf]))
    np.testing.assert_allclose(weighted.diagnostics['ppuLower'], [.5, 1.])
    np.testing.assert_allclose(weighted.diagnostics['ppuUpper'], [.5, 1.])


@pytest.mark.parametrize('weights', [[np.nan, 0], [np.inf, 0], [-np.inf, -np.inf], [0]])
def testGlueInvalidLikelihoodFailsExplicitly(weights):
    with pytest.raises(ValueError, match='likelihood|Likelihood'):
        GLUE().run(simpleProblem(), [[.2], [.8]], threshold=10.,
                   logLikelihood=lambda obs, sims, mask: np.array(weights))


def testAdaptiveCubicRejectsOvershootAndReducesFixedRmlObjective():
    problem = ModelProblem(nInput=1, lb=-100, ub=100, obs=np.array([1.0]), simFunc=lambda x:(x[:, None, :]**3).reshape(len(x), -1))
    prior = np.linspace(.05, .15, 30)[:, None]
    result = IES(maxIters=8, seed=14, adaptive=True).run(problem, prior, r=np.array([[1e-5]]))
    trials = result.diagnostics['lineSearch']
    assert any(not item['accepted'] for item in trials)
    assert any(item['step'] < 1 and item['accepted'] for item in trials)
    for item in trials:
        if item['accepted']:
            assert item['merit_after'][1] <= item['merit_before'][1] + 1e-8
    assert np.mean(result.diagnostics['scores']) < .01
    assert result.diagnostics['stopReason'] == 'step_tolerance'
    targets = 1 + np.random.default_rng(14).standard_normal(prior.shape) * np.sqrt(1e-5)
    posterior = result.posteriorDecs
    independentCost = np.mean((posterior-prior)**2 / prior.var(ddof=1) + (posterior**3-targets)**2 / 1e-5)
    assert trials[-1]['merit_after'][1] == pytest.approx(independentCost, rel=1e-12)


def testAdaptiveFailureKeepsLastAcceptedEnsembleAndWarns():
    prior = np.array([[-1.], [0.], [1.]])
    def simulate(x):
        return (np.where(np.isin(x[:, None, :], prior), x[:, None, :], 100.)).reshape(len(x), -1)
    problem = ModelProblem(nInput=1, lb=-10, ub=10, obs=np.array([2.0]), simFunc=simulate)
    with pytest.warns(RuntimeWarning, match='last accepted'):
        result = IES(adaptive=True, seed=5, maxBacktracks=3).run(problem, prior, r=np.ones((1, 1)))
    np.testing.assert_array_equal(result.posteriorDecs, prior)
    assert result.diagnostics['stopReason'] == 'line_search_stalled'
    assert len(result.diagnostics['lineSearch']) == 4
    assert not any(item['accepted'] for item in result.diagnostics['lineSearch'])


@pytest.mark.parametrize('noise', [None, np.zeros((1, 1)), np.ones((1, 1))])
def testAdaptiveLinearRetainsAnalyticFixedPoint(noise):
    prior = np.linspace(-2, 2, 30)[:, None]
    problem = ModelProblem(nInput=1, lb=-100, ub=100, obs=np.array([1.0]), simFunc=lambda x:x)
    expected = IES(maxIters=1, seed=12).run(problem, prior, r=noise)
    actual = IES(maxIters=8, seed=12, adaptive=True).run(problem, prior, r=noise)
    np.testing.assert_allclose(actual.posteriorDecs, expected.posteriorDecs, atol=1e-12)
    assert actual.diagnostics['stopReason'] == 'step_tolerance'
    assert len(actual.history.metricsHistory) <= 3


@pytest.mark.parametrize('methodClass', [ES, IES])
def testBoundaryRescalingKeepsSpreadAndReportsMomentChange(methodClass):
    def simulate(x):
        assert np.all((x >= -1) & (x <= 1))
        return x
    problem = ModelProblem(nInput=1, lb=-1, ub=1, obs=np.array([100.0]), simFunc=simulate)
    options = dict(seed=1, maxIters=1) if methodClass is IES else {}
    prior = [[-.5], [0], [.5]]
    clipped = methodClass(**options).run(problem, prior, r=np.array([[.1]]))
    scaled = methodClass(boundHandling='rescale', **options).run(problem, prior, r=np.array([[.1]]))
    assert np.ptp(clipped.posteriorDecs) == 0
    assert np.ptp(scaled.posteriorDecs) > 0
    assert np.all(scaled.posteriorDecs < 1)
    assert not scaled.diagnostics['boundEffects'][0]['unconstrained_moments_preserved']
