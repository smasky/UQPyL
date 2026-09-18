import numpy as np
import pytest
from UQPyL.calibration import ES, IES
from UQPyL.problem import ModelProblem
from UQPyL.calibration.methods._ensemble import ensembleGain, validateCovariance


def problemWithRepeatedObservations():
    def simulate(X):
        return np.repeat(X[:, None, :], 5, axis=1)
    return ModelProblem(nInput=1, lb=-10, ub=10, simFunc=simulate, obs=np.ones((5, 1)))


@pytest.mark.parametrize('methodClass', [ES, IES])
def test_default_rank_deficient_update_matches_independent_pseudoinverse(methodClass):
    X = np.array([[-1.], [0.], [2.]])
    problem = problemWithRepeatedObservations()
    Y = np.repeat(X, 5, axis=1)
    dx, dy = X-X.mean(axis=0), Y-Y.mean(axis=0)
    expected = X + (1-Y) @ ((dx.T@dy/2) @ np.linalg.pinv(dy.T@dy/2)).T
    opts = {'maxIters': 1} if methodClass is IES else {}
    result = methodClass(**opts).run(problem, X)
    np.testing.assert_allclose(result.posteriorDecs, expected, atol=1e-12)
    info = result.diagnostics['covarianceSolves'][0] if methodClass is IES else result.diagnostics['covarianceSolve']
    assert info['solver'] == 'pinv' and info['rank'] == 1 and info['dimension'] == 5


@pytest.mark.parametrize('methodClass', [ES, IES])
def test_zero_spread_is_finite_and_leaves_ensemble_unchanged(methodClass):
    X = np.zeros((3, 1))
    opts = {'maxIters': 3} if methodClass is IES else {}
    result = methodClass(**opts).run(problemWithRepeatedObservations(), X)
    np.testing.assert_array_equal(result.posteriorDecs, X)
    infos = result.diagnostics['covarianceSolves'] if methodClass is IES else [result.diagnostics['covarianceSolve']]
    assert all(info['rank'] == 0 for info in infos)


@pytest.mark.parametrize('methodClass', [ES, IES])
def test_explicit_observation_noise_uses_full_rank_solve(methodClass):
    opts = {'maxIters': 1, 'lam': .2} if methodClass is IES else {}
    result = methodClass(**opts).run(problemWithRepeatedObservations(), [[-1], [0], [2]], r=np.eye(5)*.5)
    info = result.diagnostics['covarianceSolves'][0] if methodClass is IES else result.diagnostics['covarianceSolve']
    assert info['solver'] == 'solve' and info['rank'] == 5


@pytest.mark.parametrize('r,match', [(np.eye(2), 'shape'), (np.full((5, 5), np.nan), 'finite'),
    (np.triu(np.ones((5, 5))), 'symmetric'), (-np.eye(5), 'positive semidefinite')])
@pytest.mark.parametrize('methodClass', [ES, IES])
def test_invalid_observation_covariance_is_rejected(methodClass, r, match):
    with pytest.raises(ValueError, match=match):
        methodClass().run(problemWithRepeatedObservations(), [[-1], [0], [2]], r=r)


@pytest.mark.parametrize('lam', [-1, np.nan, np.inf, [1, 2]])
def test_invalid_regularization_is_rejected(lam):
    with pytest.raises(ValueError, match='lam'):
        IES(lam=lam)


def test_full_rank_gain_matches_solve_and_does_not_mutate_covariance():
    rng = np.random.default_rng(4)
    Y = rng.normal(size=(8, 3)); X = rng.normal(size=(8, 2))
    cxy, cyy = X.T@Y, Y.T@Y
    r = np.diag([.1, .2, .3]); saved = r.copy()
    gain, info = ensembleGain(cxy, cyy, validateCovariance(r, 3), .4)
    np.testing.assert_allclose(gain, np.linalg.solve(cyy+r+.4*np.eye(3), cxy.T).T)
    np.testing.assert_array_equal(r, saved)
    assert info['solver'] == 'solve'


def test_numerically_unresolved_direction_is_discarded():
    gain, info = ensembleGain(np.eye(2), np.diag([1., 1e-20]), np.zeros((2, 2)))
    assert info['rank'] == 1
    np.testing.assert_allclose(gain, np.diag([1., 0.]))
