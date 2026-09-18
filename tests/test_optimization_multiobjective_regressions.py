from types import SimpleNamespace

import numpy as np
import pytest

from UQPyL.optimization import Population, OptReader
from UQPyL.optimization.core import NDSort
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.optimization.runtime.result import OptState
from UQPyL.optimization.runtime.verbose import MultiObjectiveRenderer, VerboseConfig
from UQPyL.problem import Problem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def makeState(ref=None, opt=1):
    algorithm = SimpleNamespace(problem=SimpleNamespace(opt=opt), FEs=0, iters=0,
                                hvRefPoint=ref)
    return OptState(algorithm)


def makePop(objs, cons=None, weights=None):
    return Population(np.arange(len(objs), dtype=float)[:, None], objs, cons, weights)


def update(state, pop, fe):
    state.algorithm.FEs = fe
    state.algorithm.iters = fe
    state.update(pop, state.algorithm.problem, fe, fe, 'MOEA')


def test_infeasible_diagnostics_are_not_a_pareto_front():
    pop = makePop([[1., 1.], [2., 2.]], [[1.], [2.]])
    assert pop.getParetoFront().objs.shape == (0, 2)
    assert len(pop.getInfeasibleCandidates()) == 2
    state = makeState()
    update(state, pop, 2)
    result = state.buildResult()
    assert not result.bestFeasible
    assert result.bestObjs.shape == (0, 2)
    assert result.bestMetric is None and state.hvRefPoint is None
    assert result.minViolation == 1
    np.testing.assert_array_equal(result.candidateCons, [[1]])
    assert result.history.numBestHistory == [0]
    assert result.history.bestMetricHistory == [None]
    assert result.history.bestObjHistory == []
    assert 'no feasible solution found' in MultiObjectiveRenderer(VerboseConfig()).renderFinal(result, 'test')


def test_first_feasible_is_improvement_and_initializes_reference():
    state = makeState()
    update(state, makePop([[1., 1.]], [[1.]]), 1)
    update(state, makePop([[10., 10.]], [[0.]]), 2)
    assert state.history.improvedHistory == [True, True]
    assert state.appearFEs == 2 and state.bestFeasible
    np.testing.assert_allclose(state.hvRefPoint, [12, 12])
    assert state.bestMetric == pytest.approx(4)
    assert state.candidates is None


def test_infeasible_progress_uses_weighted_violation_and_retains_history():
    state = makeState()
    update(state, makePop([[1., 1.]], [[2., 0.]], [10, 1]), 1)
    update(state, makePop([[100., 100.]], [[0., 3.]], [10, 1]), 2)
    update(state, makePop([[0., 0.]], [[1., 0.]], [10, 1]), 3)
    assert state.minViolation == 3 and state.appearFEs == 2
    assert state.history.improvedHistory == [True, True, False]
    np.testing.assert_array_equal(state.candidates.objs, [[100, 100]])


def test_archive_survives_regression_and_ignores_duplicates():
    state = makeState(ref=[10, 10])
    update(state, makePop([[0., 2.], [2., 0.]]), 2)
    update(state, makePop([[5., 5.]]), 3)
    update(state, makePop([[0., 2.]]), 4)
    assert state.appearFEs == 2
    assert state.history.improvedHistory == [True, False, False]
    np.testing.assert_array_equal(state.bestObjs, [[0, 2], [2, 0]])
    update(state, makePop([[1., 1.]]), 5)
    assert state.appearFEs == 5 and len(state.bestObjs) == 3
    update(state, makePop([[-1., -1.]]), 6)
    np.testing.assert_array_equal(state.bestObjs, [[-1, -1]])
    assert state.history.bestObjHistory == []  # A singleton front is still multi-objective.


def test_archive_changes_outside_hv_reference_are_still_improvements():
    state = makeState(ref=[1, 1])
    update(state, makePop([[0., 2.]]), 1)
    update(state, makePop([[2., 0.]]), 2)
    assert state.bestMetric == 0
    assert state.appearFEs == 2 and len(state.bestObjs) == 2


def test_hv_fixed_scale_preserves_strict_improvement_for_negative_objectives():
    state = makeState(ref=[10, 10])
    update(state, makePop([[-1., -1.]]), 1)
    assert state.bestMetric == pytest.approx(121)
    update(state, makePop([[-2., -2.]]), 2)
    assert state.bestMetric == pytest.approx(144)
    state = makeState()
    update(state, makePop([[-10., -10.]]), 1)
    np.testing.assert_allclose(state.hvRefPoint, [-8, -8])
    assert state.bestMetric == pytest.approx(4)


def test_reference_and_archive_exports_restore_mixed_objective_directions():
    state = makeState(ref=[0, 10], opt=np.array([[-1, 1]]))
    update(state, makePop([[-5., 2.]]), 1)
    result = state.buildResult()
    np.testing.assert_allclose(result.bestObjs, [[5, 2]])
    np.testing.assert_allclose(result.extra['hv_reference_point'], [[0, 10]])
    assert result.bestMetric == pytest.approx(40)
    result.bestObjs[:] = 99
    np.testing.assert_array_equal(state.bestObjs, [[-5, 2]])


@pytest.mark.parametrize('ref', [[1], [1, 2, 3], [np.inf, 1]])
def test_reference_validates_dimension_and_finiteness(ref):
    with pytest.raises(ValueError, match='hvRefPoint'):
        update(makeState(ref=ref), makePop([[0., 0.]]), 1)


def test_evaluated_offspring_enters_archive_before_survivor_selection():
    problem = Problem(nInput=1, nObj=2, lb=10, ub=20,
                      objFunc=lambda X: np.column_stack((X[:, 0], 30-X[:, 0])))
    alg = NSGAII(nPop=2, **QUIET)
    alg.setup(problem, 1)
    initial = alg.initPop(2, initialPop=[[10], [20]])
    alg.update(initial)
    alg.evaluate(Population([[.5]]))  # Simulate an offspring discarded by selection.
    alg.update(initial)
    result = alg.buildResult()
    np.testing.assert_allclose(result.bestDecs[:, 0], [10, 15, 20])
    assert len(alg.state.currentPop) == 2
    alg.setup(problem, 1)
    assert alg.state.archive is None


def test_equal_violation_is_same_front_and_uses_diversity():
    objs = np.array([[1., 1.], [0., 2.], [2., 0.]])
    cons = np.ones((3, 1))
    ranks, last = NDSort(objs, cons, nSort=2)
    np.testing.assert_array_equal(ranks, [1, 1, 1])
    assert last == 1
    keep, _, _ = NSGAII(**QUIET).environmentalSelection(np.arange(3)[:, None], objs, cons, None, 2)
    np.testing.assert_array_equal(np.flatnonzero(keep), [1, 2])


@pytest.mark.parametrize('aggregation', ['PBI', 'TCH', 'TCH_N', 'TCH_M'])
def test_moead_small_population_finishes_iteration_at_budget_boundary(aggregation):
    problem = Problem(nInput=1, nObj=2, nCon=1, lb=0, ub=1,
                      objFunc=lambda X: np.zeros((len(X), 2)),
                      conFunc=lambda X: -np.ones((len(X), 1)))
    algorithm = MOEAD(aggregation=aggregation, nPop=8, maxFEs=19, maxIters=10, **QUIET)
    with np.errstate(divide='raise', invalid='raise'):
        result = algorithm.run(problem, seed=2)
    assert result.FEs == 24
    assert result.iters == 2
    assert result.bestFeasible and result.bestMetric > 0


def test_rvea_singleton_keeps_reference_scale_and_diverse_feasible_solutions():
    alg = RVEA(**QUIET)
    vectors = np.array([[1., 0.], [.5, .5], [0., 1.]])
    scaled = alg.updateReferenceVector(np.array([[1., 1.]]), vectors)
    np.testing.assert_array_equal(scaled, vectors)
    objs = np.array([[0., 2.], [1., 1.], [2., 0.]])
    with np.errstate(divide='raise', invalid='raise'):
        keep = alg.environmentSelection(objs, scaled, .5, np.zeros((3, 1)))
    assert set(keep) == {0, 1, 2}
    alg.updateReferenceVector(np.array([[0., 0.], [2., 3.]]), vectors)
    scaled = alg.updateReferenceVector(np.array([[1., 1.]]), vectors)
    np.testing.assert_allclose(scaled, vectors * [2, 3])


@pytest.mark.parametrize('algorithmClass', [NSGAIII, RVEA])
def test_reference_selection_is_finite_with_constant_objectives(algorithmClass):
    algorithm = algorithmClass(**QUIET)
    algorithm.rng = np.random.default_rng(3)
    objs = np.zeros((4, 2))
    vectors = np.array([[1., 0.], [0., 1.]])
    with np.errstate(divide='raise', invalid='raise'):
        if algorithmClass is NSGAIII:
            keep, _ = algorithm.environmentSelection(objs, np.zeros((4, 1)), vectors, np.zeros((1, 2)))
            assert keep.sum() == 2
        else:
            keep = algorithm.environmentSelection(objs, vectors, .5, np.zeros((4, 1)))
            assert len(keep) >= 1


def test_empty_pareto_and_candidates_roundtrip_sqlite_npz(tmp_path):
    problem = Problem(nInput=1, nObj=2, nCon=1, lb=10, ub=20,
                      objFunc=lambda X: np.column_stack((X[:, 0], 30-X[:, 0])),
                      conFunc=lambda X: np.ones((len(X), 1)), conWgt=[3])
    problem.workDir = str(tmp_path)
    algorithm = NSGAII(nPop=4, maxFEs=8, verboseFlag=False, logFlag=False, saveFlag=True)
    result = algorithm.run(problem, seed=1)
    db = next(tmp_path.rglob('*.sqlite3'))
    reader = OptReader(db)
    try:
        assert reader.load_last_best().objs.shape == (0, 2)
        candidate = reader.load_last_candidates()
        np.testing.assert_allclose(candidate.decs, result.candidateDecs)
        np.testing.assert_allclose(candidate.objs, result.candidateObjs)
        np.testing.assert_allclose(candidate.conWgt, [[3]])
        snapshot = reader.list_snapshots()[-1]
        assert snapshot['paretoSize'] == 0
        assert snapshot['hypervolume'] is None
        assert snapshot['constraintViolation'] == 3
    finally:
        reader.close()
    payload = algorithm.saveResult()
    np.testing.assert_allclose(payload['candidate_decs'], result.candidateDecs)
    assert payload['bestObjs'].shape == (0, 2)
    assert payload['min_violation'] == 3


@pytest.mark.parametrize('algorithmClass', [NSGAII, NSGAIII, MOEAD, RVEA])
def test_complete_constrained_runs_archive_every_evaluation(algorithmClass):
    evaluated = []
    def objective(X):
        evaluated.extend(X[:, 0].tolist())
        return np.column_stack((X[:, 0], 1-X[:, 0]))
    problem = Problem(nInput=1, nObj=2, nCon=1, lb=0, ub=1,
                      objFunc=objective, conFunc=lambda X: .4-X)
    algorithm = algorithmClass(nPop=8, maxFEs=24, maxIters=5, **QUIET)
    result = algorithm.run(problem, seed=5)
    expected = np.unique(np.asarray(evaluated)[np.asarray(evaluated) >= .4])
    np.testing.assert_allclose(result.bestDecs[:, 0], expected)
    assert np.all(result.bestCons <= 0)
    assert result.history.numBestHistory == sorted(result.history.numBestHistory)


def test_feasible_archive_roundtrip_with_original_maximization_values(tmp_path):
    problem = Problem(nInput=1, nObj=2, nCon=1, lb=10, ub=20,
                      objFunc=lambda X: np.column_stack((X[:, 0], X[:, 0])),
                      conFunc=lambda X: 12-X, optType=['max', 'min'])
    problem.workDir = str(tmp_path)
    algorithm = NSGAII(nPop=4, maxFEs=8, hvRefPoint=[0, 30],
                      verboseFlag=False, logFlag=False, saveFlag=True)
    result = algorithm.run(problem, seed=1)
    reader = OptReader(next(tmp_path.rglob('*.sqlite3')))
    try:
        loaded = reader.load_last_best()
        np.testing.assert_array_equal(reader.load_algorithm().hvRefPoint, [0, 30])
        np.testing.assert_allclose(loaded.decs, result.bestDecs)
        np.testing.assert_allclose(loaded.objs, result.bestObjs)
        assert len(reader.load_last_candidates()) == 0
        assert reader.list_snapshots()[-1]['hypervolume'] == pytest.approx(result.bestMetric)
    finally:
        reader.close()
    assert result.bestMetric > 0
    np.testing.assert_allclose(result.bestObjs, np.repeat(result.bestDecs, 2, axis=1))


def test_preevaluated_initial_members_enter_archive_without_counting_evaluations():
    problem = Problem(nInput=1, nObj=2, lb=10, ub=20,
                      objFunc=lambda X: np.column_stack((X[:, 0], 30-X[:, 0])))
    algorithm = NSGAII(nPop=2, **QUIET)
    algorithm.setup(problem, 1)
    initial = Population([[10.], [20.]], [[10., 20.], [20., 10.]])
    pop = algorithm.initPop(2, initialPop=initial)
    algorithm.update(pop)
    assert algorithm.FEs == 0 and algorithm.state.appearFEs == 0
    np.testing.assert_array_equal(algorithm.buildResult().bestDecs, initial.decs)


def test_candidate_result_and_history_restore_directions_without_shared_arrays():
    state = makeState(opt=np.array([[-1, 1]]))
    update(state, makePop([[-5., 2.]], [[1.]]), 1)
    result = state.buildResult()
    np.testing.assert_array_equal(result.candidateObjs, [[5, 2]])
    np.testing.assert_array_equal(result.history.bests[0]['candidateObjs'], [[5, 2]])
    result.candidateObjs[:] = 99
    result.history.bests[0]['candidateObjs'][:] = 99
    np.testing.assert_array_equal(state.candidates.objs, [[-5, 2]])
    np.testing.assert_array_equal(state.history.bests[0]['candidateObjs'], [[-5, 2]])
