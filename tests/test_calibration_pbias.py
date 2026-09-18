import numpy as np
import pytest

from UQPyL.calibration import GLUE, SUFI2, ES, IES
from UQPyL.calibration.util import pbias
from UQPyL.problem import ModelProblem


def biasProblem():
    obs = np.array([[10.], [20.], [30.]])
    def simulate(X):
        return np.broadcast_to(obs[None, :, :]+X[:, None, :], (len(X), 3, 1)).copy()
    return ModelProblem(nInput=1, lb=-20, ub=20, obs=obs, simFunc=simulate)


def test_pbias_retains_sign_and_sums_before_comparison_absolute_value():
    obs = np.array([10., 20.])
    np.testing.assert_allclose(pbias(obs, [[9, 18], [11, 22], [9, 21]]), [-10, 10, 0])


@pytest.mark.parametrize('threshold,expected', [(0, [0]), (5, [-1, 0, 1]), (30, [-6, -1, 0, 1, 6])])
def test_glue_symmetric_inclusive_threshold_and_raw_diagnostics(threshold, expected):
    X = np.array([[-6.], [-1.], [0.], [1.], [6.]])
    result = GLUE(metric='pbias').run(biasProblem(), X, threshold=threshold)
    np.testing.assert_allclose(result.bestDecs, [[0]])
    np.testing.assert_allclose(result.behavioralDecs[:, 0], expected)
    np.testing.assert_allclose(result.diagnostics['scores'], [-30, -5, 0, 5, 30])
    np.testing.assert_allclose(result.diagnostics['behavioralScores'], np.array(expected)*5)


@pytest.mark.parametrize('threshold', [-1, np.nan, np.inf])
def test_glue_rejects_invalid_pbias_tolerance(threshold):
    with pytest.raises(ValueError, match='PBIAS threshold'):
        GLUE(metric='pbias').run(biasProblem(), [[0]], threshold=threshold)


def test_glue_rejects_large_negative_bias_and_preserves_callable_semantics():
    with pytest.raises(ValueError, match='No behavioral'):
        GLUE(metric='pbias').run(biasProblem(), [[-6]], threshold=5)
    # The explicit label enables the special rule, not the function's name.
    result = GLUE(metric=pbias).run(biasProblem(), [[-6], [0]], threshold=5)
    np.testing.assert_array_equal(result.bestDecs, [[-6]])


def test_sufi2_ranks_absolute_bias_but_keeps_signed_elite_scores():
    result = SUFI2(metric='pbias').run(biasProblem(), [[-6], [-1], [0], [1], [6]], eliteSize=3)
    np.testing.assert_allclose(result.bestDecs, [[0]])
    assert set(result.eliteDecs[:, 0]) == {-1, 0, 1}
    assert set(result.diagnostics['eliteScores']) == {-5, 0, 5}


@pytest.mark.parametrize('methodClass', [ES, IES])
def test_smoothers_choose_nearest_zero_and_keep_raw_bias(methodClass):
    options = dict(maxIters=2, lam=1e-6) if methodClass is IES else {}
    result = methodClass(metric='pbias', **options).run(
        biasProblem(), [[-6], [-1], [0], [1], [6]], r=np.eye(3)*3)
    scores = result.diagnostics['scores']
    assert scores.min() < 0 < scores.max()
    best = np.argmin(abs(scores))
    assert result.extra['bestIdx'] == best
    np.testing.assert_allclose(result.bestDecs, result.posteriorDecs[best:best+1])
    np.testing.assert_allclose(scores, pbias(np.array([10, 20, 30]), result.posteriorSims))
