import numpy as np
import pytest

from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS, InfReader
from UQPyL.problem import Problem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)
METHODS = [MH, MH_Gibbs, AMH, DEMC, DREAM_ZS]


def mixedProblem(constrained=False, seen=None):
    def objective(X):
        assert np.all(np.isin(X[:, 0], [10, 20, 30]))
        assert np.all(np.isin(X[:, 1], [1, 2, 3]))
        if seen is not None: seen.extend(X.copy())
        return (X[:, [0]]/30 + X[:, [1]]*.05 + X[:, [2]]*.01)
    return Problem(nInput=3, nObj=1, nCon=int(constrained), lb=[0, 1, -2], ub=[1, 3, 2],
                   varType=[2, 1, 0], varSet={0: [10, 20, 30]}, objFunc=objective,
                   conFunc=(lambda X: X[:, [0]]-20) if constrained else None)


@pytest.mark.parametrize('methodClass', METHODS)
@pytest.mark.parametrize('constrained', [False, True])
def test_mixed_chains_match_actual_evaluations_and_custom_log_probability(methodClass, constrained):
    seen = []
    problem = mixedProblem(constrained, seen)
    logCalls = []
    def logProbability(y, decs, cons):
        decs = np.atleast_2d(decs)
        assert np.all(np.isin(decs[:, 0], [10, 20, 30]))
        assert np.all(np.isin(decs[:, 1], [1, 2, 3]))
        expected = decs[:, 0]/30 + decs[:, 1]*.05 + decs[:, 2]*.01
        np.testing.assert_allclose(np.ravel(y), expected)
        logCalls.append(decs.copy())
        return -expected
    method = methodClass(nChains=4, warmUp=2, logProbFunc=logProbability,
                         **{("maxIterTimes" if methodClass in (AMH, DEMC) else "maxIters"): 12}, **QUIET)
    result = method.run(problem, seed=17)
    recorded = result.decs.reshape(-1, 3)
    assert logCalls
    for row in recorded:
        assert any(np.array_equal(row, evaluated) for evaluated in seen)
    expected = problem.evaluate(recorded)
    np.testing.assert_allclose(result.objs.reshape(-1, 1), expected.objs)
    np.testing.assert_allclose(result.logProb.reshape(-1), -expected.objs[:, 0])
    if constrained:
        assert np.all(result.cons <= 0)
        np.testing.assert_allclose(result.cons.reshape(-1, 1), expected.cons)
    np.testing.assert_allclose(result.bestObjs, problem.evaluate(result.bestDecs).objs)


def test_decoding_does_not_snap_or_modify_latent_proposals():
    problem = mixedProblem()
    method = MH(nChains=3, maxIters=1, **QUIET)
    method.setup(problem, seed=1)
    latent = np.array([[.1, 1.1, -.7], [.4, 1.8, .3], [.9, 2.9, 1.2]])
    saved = latent.copy()
    expected = np.array([[10., 1., -.7], [20., 2., .3], [30., 3., 1.2]])
    np.testing.assert_array_equal(method._decodeDecs(latent), expected)
    values, _ = method.evaluate(latent)
    np.testing.assert_allclose(values, problem.evaluate(expected).objs)
    np.testing.assert_array_equal(latent, saved)


@pytest.mark.parametrize('methodClass', METHODS)
def test_fixed_continuous_dimension_survives_reflection(methodClass):
    problem = Problem(nInput=2, nObj=1, lb=[0, 5], ub=[1, 5], varType=[2, 0], varSet={0: [10, 20]},
                      objFunc=lambda X: X[:, [0]]/20)
    with np.errstate(divide='raise', invalid='raise'):
        result = methodClass(nChains=4, warmUp=1,
                             **{("maxIterTimes" if methodClass in (AMH, DEMC) else "maxIters"): 5}, **QUIET).run(problem, seed=3)
    assert np.all(result.decs[..., 1] == 5)
    assert np.all(np.isin(result.decs[..., 0], [10, 20]))


def test_discrete_result_and_snapshot_roundtrip(tmp_path):
    problem = mixedProblem(True)
    problem.workDir = str(tmp_path)
    method = MH(nChains=3, warmUp=1, maxIters=5, verboseFlag=False, logFlag=False, saveFlag=True)
    result = method.run(problem, seed=3)
    with InfReader(next(tmp_path.rglob('*.sqlite3'))) as reader:
        loaded = reader.load_result()
        np.testing.assert_array_equal(loaded.decs, result.decs)
        np.testing.assert_array_equal(loaded.objs, result.objs)
        members = reader.load_last_snapshot_members()
        for member in members:
            assert np.asarray(member['decs']).ravel()[0] in (10, 20)


def test_mh_discrete_marginal_matches_target_masses():
    weights = np.array([.2, .3, .5])
    def objective(X):
        return -np.log(weights[(X[:, 0]/10-1).astype(int)])[:, None]
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, varType=[2], varSet={0: [10, 20, 30]}, objFunc=objective)
    result = MH(nChains=4, warmUp=100, maxIters=1200, **QUIET).run(problem, gamma=.6, seed=13)
    frequencies = np.array([np.mean(result.decs == value) for value in (10, 20, 30)])
    np.testing.assert_allclose(frequencies, weights, atol=.05)


def test_integer_values_occupy_equal_latent_widths():
    problem = Problem(nInput=1, nObj=1, lb=1, ub=3, varType=[1], objFunc=lambda X: X)
    method = MH(**QUIET)
    method.setup(problem, 1)
    latent = (1 + (np.arange(300)+.5)/300*2)[:, None]
    real = method._decodeDecs(latent)
    np.testing.assert_array_equal(np.unique(real, return_counts=True)[1], [100, 100, 100])
