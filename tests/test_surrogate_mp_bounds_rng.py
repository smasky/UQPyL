from types import SimpleNamespace

import numpy as np
import pytest

from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.util.boxmin import Boxmin
from UQPyL.surrogate.util.lbfgsb import LBFGSB


@pytest.mark.parametrize("optimizerClass", [Boxmin, LBFGSB])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("lb,ub,initial", [
    ([-3.0], [-1.0], [-3.0]),
    ([-2.0], [3.0], [0.0]),
    ([0.0], [4.0], [0.0]),
    ([1.0], [5.0], [5.0]),
    ([-3.0, 2.0, -1.0], [-1.0, 2.0, 4.0], [7.0, 9.0, -8.0]),
    ([2.0], [2.0], [9.0]),
])
def test_every_evaluation_and_returned_point_is_bounded(optimizerClass, direction, lb, ub, initial):
    lower, upper = np.array(lb), np.array(ub)
    start = np.array(initial)
    original = start.copy()
    evaluations = []

    def objective(X):
        X = np.asarray(X).reshape(-1)
        assert np.all(X >= lower) and np.all(X <= upper)
        evaluations.append(X.copy())
        return float(direction * np.sum(X))

    problem = SimpleNamespace(lb=lower, ub=upper, nInput=len(lower), objFunc=objective)
    optimizer = optimizerClass()
    best, score = optimizer.run(problem, xInit=start, seed=2)
    assert np.all(best >= lower) and np.all(best <= upper)
    np.testing.assert_allclose(best, lower if direction == 1 else upper, atol=1e-6)
    assert score == pytest.approx(direction * np.sum(best))
    np.testing.assert_array_equal(start, original)
    if optimizerClass is Boxmin:
        assert optimizer.nv == len(evaluations)


@pytest.mark.parametrize("optimizerClass", [Boxmin, LBFGSB])
@pytest.mark.parametrize("explicitStart", [False, True])
@pytest.mark.parametrize("seed", [None, 5])
def test_internal_optimizer_does_not_change_global_rng(optimizerClass, explicitStart, seed):
    problem = SimpleNamespace(lb=np.array([-3.0]), ub=np.array([2.0]), nInput=1,
                              objFunc=lambda X: float(np.sum((X + 0.4)**2)))
    before = np.random.get_state()
    try:
        optimizerClass().run(problem, xInit=np.array([0.0]) if explicitStart else None, seed=seed)
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(before)


@pytest.mark.parametrize("optimizerClass", [Boxmin, LBFGSB])
def test_seed_reproduces_entire_evaluation_sequence(optimizerClass):
    traces = []
    results = []
    optimizer = optimizerClass()
    for _ in range(2):
        trace = []

        def objective(X):
            trace.append(np.asarray(X).copy())
            return float(np.sum((X - 0.2)**2))

        problem = SimpleNamespace(lb=np.array([-2.0, 0.0]), ub=np.array([3.0, 5.0]),
                                  nInput=2, objFunc=objective)
        results.append(optimizer.run(problem, seed=42))
        traces.append(trace)
    np.testing.assert_array_equal(traces[0], traces[1])
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_boxmin_can_move_away_from_zero_towards_interior_optimum():
    problem = SimpleNamespace(lb=np.array([-2.0]), ub=np.array([3.0]), nInput=1,
                              objFunc=lambda X: float((X[0] - 1.0)**2))
    _, score = Boxmin().run(problem, xInit=np.array([0.0]))
    assert score < 0.1


@pytest.mark.parametrize("lower,upper,nInputs", [
    ([2.0], [1.0], 1),
    ([0.0], [np.inf], 1),
    ([np.nan], [1.0], 1),
    ([0.0], [1.0], 2),
    ([-1e308], [1e308], 1),
])
def test_boxmin_rejects_invalid_bounds_before_evaluation(lower, upper, nInputs):
    def objective(X):
        pytest.fail("Invalid bounds must fail before calling the objective.")

    problem = SimpleNamespace(lb=np.array(lower), ub=np.array(upper), nInput=nInputs, objFunc=objective)
    with pytest.raises(ValueError):
        Boxmin().run(problem)


@pytest.mark.parametrize("initial", [[np.nan], [np.inf], [1.0, 2.0]])
def test_boxmin_rejects_invalid_initial_points(initial):
    problem = SimpleNamespace(lb=np.array([-1.0]), ub=np.array([1.0]), nInput=1,
                              objFunc=lambda X: pytest.fail("Invalid initial point reached the objective."))
    with pytest.raises(ValueError, match="xInit"):
        Boxmin().run(problem, xInit=initial)


def test_boxmin_search_is_invariant_to_affine_parameter_coordinates():
    results = []
    for lower, span in [(np.zeros(2), np.ones(2)), (np.array([-4.0, 2.0]), np.array([8.0, 16.0]))]:
        def objective(X):
            unit = (X - lower) / span
            return float(np.sum((unit - [0.3, 0.6])**2))

        problem = SimpleNamespace(lb=lower, ub=lower + span, nInput=2, objFunc=objective)
        best, score = Boxmin().run(problem, xInit=lower + span * [0.25, 0.75])
        results.append(((best - lower) / span, score))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-12)
    assert results[0][1] == pytest.approx(results[1][1], abs=1e-12)
    assert results[0][1] < 1e-4


def assertSettingsBounded(model):
    names = model.getParaList()
    _, upper, lower = model.setting.getParaInfos(names)
    values = []
    for name in names:
        value = np.atleast_1d(model.setting.get(name)).astype(float)
        values.extend(np.log(value) if model.setting.parLog[name] else value)
    values = np.array(values)
    assert np.all(values >= lower - 1e-12) and np.all(values <= upper + 1e-12)


class CheckedGPR(GPR):
    def _objfunc(self, X, Y, record=False):
        assertSettingsBounded(self)
        return super()._objfunc(X, Y, record=record)


class CheckedKRG(KRG):
    def _objFunc(self, Y, F, D, record=False):
        assertSettingsBounded(self)
        return super()._objFunc(Y, F, D, record=record)


@pytest.mark.parametrize("modelClass", [CheckedGPR, CheckedKRG])
@pytest.mark.parametrize("optimizer", ["Boxmin", "LBFGSB"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_real_model_tuning_stays_bounded_and_preserves_global_rng(modelClass, optimizer, seed):
    X = np.linspace(0, 1, 16).reshape(-1, 1)
    Y = np.sin(6 * X)
    model = modelClass(optimizer=optimizer)
    model.rng = np.random.default_rng(seed)
    before = np.random.get_state()
    try:
        model.fit(X, Y)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        assert before[0] == after[0] and before[2:] == after[2:]
        assertSettingsBounded(model)
        mean, variance = model.predict(X, returnVar=True)
        assert np.isfinite(mean).all() and np.isfinite(variance).all()
    finally:
        np.random.set_state(before)
