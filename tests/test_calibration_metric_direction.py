import numpy as np
import pytest

from UQPyL.calibration import ES, IES
from UQPyL.calibration import util
from UQPyL.problem import ModelProblem


@pytest.mark.parametrize('methodClass', [ES, IES])
@pytest.mark.parametrize('metric,higher', [('rmse', False), ('nse', True), ('kge', True)])
def test_best_member_follows_metric_direction_and_preserves_raw_scores(methodClass, metric, higher):
    obs = np.array([[1.], [2.], [4.]])
    def simulate(X):
        return np.column_stack((X[:, 0], X[:, 1]**2, X[:, 0]+X[:, 1]))[:, :, None]
    problem = ModelProblem(nInput=2, lb=0, ub=4, obs=obs, simFunc=simulate)
    ensemble = np.array([[.2, .5], [1., 1.5], [2., 2.], [3., .8]])
    options = dict(maxIters=2, lam=1e-6) if methodClass is IES else {}
    method = methodClass(metric=metric, verboseFlag=False, **options)
    result = method.run(problem, ensemble, r=np.eye(3)*.5)
    raw = getattr(util, metric)(obs.ravel(), result.posteriorSims)
    assert np.all(np.isfinite(raw)) and np.ptp(raw) > 1e-4
    expected = int(np.argmax(raw) if higher else np.argmin(raw))
    assert result.extra['bestIdx'] == expected
    np.testing.assert_allclose(result.bestDecs, result.posteriorDecs[expected:expected+1])
    np.testing.assert_allclose(result.bestSim, result.posteriorSims[expected:expected+1])
    np.testing.assert_allclose(result.diagnostics['scores'], raw)
    np.testing.assert_allclose(method.normalizedScore(result.posteriorSims), -raw if higher else raw)
    # Metric choice governs ranking, not the ensemble update equation.
    baseline = methodClass(metric='rmse', verboseFlag=False, **options).run(
        problem, ensemble, r=np.eye(3)*.5)
    np.testing.assert_allclose(result.posteriorDecs, baseline.posteriorDecs)
