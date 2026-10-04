"""Invalid densities, support initialization, snooker support and incremental moments."""
import numpy as np
import pytest

from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS, InfReader
from UQPyL.inference.chain import Chain
from UQPyL.problem import Problem


methods = [MH, MH_Gibbs, AMH, DEMC, DREAM_ZS]
quiet = dict(saveFlag=False, verboseFlag=False, logFlag=False)


def flatProblem(dimension=2, lower=-1., upper=1.):
    return Problem(nInput=dimension, nObj=1, lb=lower, ub=upper,
                   objFunc=lambda x: np.zeros((len(x), 1)))


@pytest.mark.parametrize("methodClass", methods)
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def testInvalidLogProbabilityCannotProduceASuccessfulResult(methodClass, invalid):
    def logProbability(y, decs=None, cons=None):
        return np.full(len(np.atleast_2d(decs)), invalid)
    method = methodClass(nChains=4, warmUp=0, maxIters=4, logProbFunc=logProbability, **quiet)
    with pytest.raises(ValueError, match="NaN or \\+inf"):
        method.run(flatProblem(), seed=17)


@pytest.mark.parametrize("returned", [0., [0.], [[0., 0., 0., 0.]], np.zeros((4, 2)),
                                     [0., 0., 0., 0., 0.], [1j]*4, ["0"]*4])
def testMalformedBatchLogProbabilityIsRejected(returned):
    method = MH(nChains=4, warmUp=0, maxIters=4, logProbFunc=lambda *a, **kw: returned, **quiet)
    with pytest.raises(ValueError, match="log probability"):
        method.run(flatProblem(), seed=17)


@pytest.mark.parametrize("shape", ["vector", "column", "single_scalar"])
def testAcceptedLogProbabilityShapes(shape):
    def logProbability(y, decs=None, cons=None):
        count = len(np.atleast_2d(decs))
        if shape == "single_scalar" and count == 1:
            return 0.
        return np.zeros((count, 1)) if shape == "column" else np.zeros(count)
    result = MH(nChains=4, warmUp=0, maxIters=4, logProbFunc=logProbability, **quiet).run(flatProblem(), seed=17)
    np.testing.assert_array_equal(result.logProb, np.zeros((4, 4)))


@pytest.mark.parametrize("methodClass", methods)
def testZeroProbabilityRegionIsResampledInitiallyAndRejectedLater(methodClass):
    def logProbability(y, decs=None, cons=None):
        return np.where(np.atleast_2d(decs)[:, 0] >= .4, 0., -np.inf)
    method = methodClass(nChains=4, warmUp=3, maxIters=30, logProbFunc=logProbability, **quiet)
    result = method.run(flatProblem(), seed=17)
    assert np.all(result.decs[..., 0] >= .4)
    assert np.all(np.isfinite(result.logProb))
    assert not np.all(result.accepted[:, 1:])


def testEmptySupportStopsAtInitializationBudget():
    seen = []
    def logProbability(y, decs=None, cons=None):
        seen.append(len(np.atleast_2d(decs)))
        return np.full(seen[-1], -np.inf)
    method = MH(nChains=4, maxInitAttempts=3, warmUp=0, maxIters=4, logProbFunc=logProbability, **quiet)
    with pytest.raises(ValueError, match="positive-probability.*3 LHS batches"):
        method.run(flatProblem(), seed=17)
    assert method.FEs == 12 and seen == [4, 4, 4]


def testTwoZeroProbabilityStatesNeverSubtractInfinities():
    method = MH(**quiet)
    method.setup(flatProblem(), seed=17)
    with np.errstate(all="raise"):
        assert not method.accept([np.inf], [np.inf])
        assert not method.accept([np.inf], [0.])
        assert method.accept([0.], [np.inf])


def testInvalidProbabilityDuringProposalStopsAndClosesSavedRun(tmp_path):
    calls = [0]
    def logProbability(y, decs=None, cons=None):
        calls[0] += 1
        # Valid initialization, invalid first single-state acceptance evaluation.
        count = len(np.atleast_2d(decs))
        return np.full(count, np.nan if count == 1 else 0.)
    problem = flatProblem()
    problem.workDir = str(tmp_path)
    method = MH(nChains=4, warmUp=0, maxIters=4, verboseFlag=False, saveFlag=True, logProbFunc=logProbability)
    with pytest.raises(ValueError, match="NaN"):
        method.run(problem, seed=17)
    assert method.session is None
    with InfReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        assert reader.get_run()["status"] == "failed"


def testSnookerSmallArchiveExploresAllActiveDirectionsAndReportsRefresh(tmp_path):
    problem = flatProblem(4)
    problem.workDir = str(tmp_path)
    method = DREAM_ZS(nChains=3, archSize=1, ps=1., warmUp=20, maxIters=100, verboseFlag=False, saveFlag=True)
    with pytest.warns(RuntimeWarning, match="10% full-dimensional Gaussian refresh"):
        result = method.run(problem, seed=17)
    flat = result.decs.reshape(-1, 4)
    assert np.linalg.matrix_rank(flat-flat.mean(0)) == 4
    settings = result.diagnostics["sampler"]["proposal_settings"]
    assert settings["snooker_probability"] == 1.
    assert settings["full_support_refresh_probability"] == .1
    assert settings["effective_snooker_probability"] == .9
    with InfReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        assert reader.load_result().diagnostics == result.diagnostics


def testSnookerRefreshRespectsFixedAxesAndUnits():
    traces = []
    for span in [np.array([1., 1., 0.]), np.array([.01, 100., 0.])]:
        problem = flatProblem(3, np.array([0., 0., 7.]), np.array([0., 0., 7.])+span)
        method = DREAM_ZS(nChains=3, archSize=1, ps=1., snookerRefreshProb=.25, warmUp=4, maxIters=80, **quiet)
        with pytest.warns(RuntimeWarning, match="25%"):
            result = method.run(problem, seed=17)
        np.testing.assert_array_equal(result.decs[..., 2], 7.)
        traces.append(result.decs[..., :2]/span[:2])
    np.testing.assert_allclose(traces[0], traces[1], atol=1e-12, rtol=0)


@pytest.mark.parametrize("probability", [0., -.1, 1.1, np.nan])
def testRefreshProbabilityMustProvideFullSupport(probability):
    with pytest.raises(ValueError, match="snookerRefreshProb"):
        DREAM_ZS(snookerRefreshProb=probability, **quiet).run(flatProblem(), seed=17)


@pytest.mark.parametrize("dimension", [1, 4])
@pytest.mark.parametrize("offset", [0., 1e12])
def testIncrementalCovarianceMatchesIndependentCenteredMoments(dimension, offset):
    rng = np.random.default_rng(17)
    samples = offset + rng.normal(size=(120, dimension))
    samples[::4] = samples[0]  # Include repeated occupation states.
    chain = Chain(dimension, 1, 0, len(samples))
    method = AMH(**quiet)
    method.setProblem(flatProblem(dimension, offset-10., offset+10.))
    for index, point in enumerate(samples):
        chain.add(point, [0.])
        if index < 2:
            continue
        covariance = method.updateCovs([chain], 2.)[0]
        precise = samples[:index+1].astype(np.longdouble)
        centered = precise-precise[0]
        centered -= centered.mean(axis=0)
        expected = np.asarray(centered.T @ centered/index, dtype=float)*2 + np.eye(dimension)*.4
        np.testing.assert_allclose(covariance, expected, rtol=2e-14, atol=2e-14)
        covariance[:] = -999.  # Returned matrices cannot corrupt cached scatter.


def testIncrementalCovarianceSkipsOldRowsAndResetsCleanly():
    samples = np.random.default_rng(17).normal(size=(12, 2))
    chain = Chain(2, 1, 0, len(samples))
    method = AMH(**quiet)
    method.setup(flatProblem(2, -10., 10.), seed=17)
    for point in samples[:5]:
        chain.add(point, [0.])
    method.updateCovs([chain], 1.)
    class AppendOnlyView(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, slice):
                assert key.start == 5, "Old history was rescanned"
            return super().__getitem__(key)
    chain.decs = chain.decs.view(AppendOnlyView)
    for point in samples[5:]:
        chain.add(point, [0.])
    actual = method.updateCovs([chain], 1.)[0]
    expected = np.cov(samples.T) + np.eye(2)*.4
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    method.reset()
    assert len(method._covarianceMoments) == 0
