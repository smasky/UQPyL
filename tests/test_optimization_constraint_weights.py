import numpy as np
import pytest

from UQPyL.problem import Problem
from UQPyL.optimization import Population, OptReader
from UQPyL.optimization.soea import GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.optimization.core import NDSort
from UQPyL.optimization.core.constraint import calcConstraintViolation
from UQPyL.optimization.runtime.verbose import _constraint_violation

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def problemWithWeights(nObj=1, weights=None):
    return Problem(nInput=2, nObj=nObj, nCon=2, lb=10, ub=20,
        conWgt=[10, 1] if weights is None else weights,
        objFunc=lambda X: np.column_stack((X[:, 0], X[:, 1]))[:, :nObj],
        conFunc=lambda X: np.column_stack((2*(20-X[:, 0])/10, 3*(X[:, 0]-10)/10)))


@pytest.mark.parametrize('weights', [[-1, 1], [np.nan, 1], [np.inf, 1], [1], [1, 2, 3], [[1, 2]]])
def test_problem_rejects_invalid_constraint_weights(weights):
    with pytest.raises(ValueError, match='conWgt'):
        problemWithWeights(weights=weights)


def test_weighted_violation_formula_and_broadcast_validation():
    assert calcConstraintViolation([[2, -3, 4]], [10, 1, .5])[0] == 22
    assert _constraint_violation([[2, -3, 4]], [10, 1, .5]) == 22
    with pytest.raises(ValueError, match='conWgt'):
        calcConstraintViolation([[2, 3]], [10])


def test_problem_weights_override_initial_population_and_are_copied():
    problem = problemWithWeights()
    real = np.array([[10., 10.], [20., 20.]])
    evaluation = problem.evaluate(real)
    initial = Population(real, evaluation.objs, evaluation.cons, conWgt=[1, 1])
    algorithm = GA(nPop=2, maxFEs=2, **QUIET)
    algorithm.setup(problem, 4)
    pop = algorithm.initPop(2, initialPop=initial)
    assert algorithm.FEs == 0
    np.testing.assert_array_equal(pop.conWgt, [[10, 1]])
    np.testing.assert_array_equal(initial.conWgt, [[1, 1]])
    assert not np.shares_memory(pop.conWgt, problem.conWgt)
    assert pop.getBest().objs[0, 0] == 20
    algorithm.updateState(pop)
    assert algorithm.buildResult().bestDecs[0, 0] == 20


def test_population_slice_best_merge_replace_keep_independent_weights():
    pop = Population([[0], [1]], [[0], [100]], [[2, 0], [0, 3]], conWgt=[10, 1])
    for derived in [pop.copy(), pop[:1], pop.getBest(), pop.merged(pop)]:
        np.testing.assert_array_equal(derived.conWgt, [[10, 1]])
        assert not np.shares_memory(derived.conWgt, pop.conWgt)
    replacement = pop[1:]
    pop.replace(0, replacement)
    np.testing.assert_array_equal(pop.conWgt, [[10, 1]])
    assert pop.getBest().objs[0, 0] == 100


SO_FACTORIES = [lambda cls=cls: cls(nPop=8, maxFEs=24, maxIters=1, **QUIET)
                for cls in [GA, DE, PSO, ABC, CSA]]
SO_FACTORIES += [lambda cls=cls: cls(ngs=2, maxFEs=24, maxIters=1, **QUIET)
                 for cls in [SCE_UA, ML_SCE_UA]]
MO_FACTORIES = [lambda cls=cls: cls(nPop=8, maxFEs=24, maxIters=1, **QUIET)
                for cls in [NSGAII, NSGAIII, MOEAD, RVEA]]


@pytest.mark.parametrize('factory,nObj', [(f, 1) for f in SO_FACTORIES]+[(f, 2) for f in MO_FACTORIES])
def test_algorithm_offspring_and_history_keep_problem_weights(factory, nObj):
    problem = problemWithWeights(nObj)
    algorithm = factory()
    originalEvaluate = algorithm.evaluate
    calls = []
    def evaluate(pop):
        originalEvaluate(pop)
        np.testing.assert_array_equal(pop.conWgt, problem.conWgt)
        calls.append(len(pop))
        return pop
    algorithm.evaluate = evaluate
    result = algorithm.run(problem, seed=3)
    assert len(calls) > 1
    np.testing.assert_array_equal(result.extra['constraint_weights'], problem.conWgt)
    for snapshot in result.history.populations:
        np.testing.assert_array_equal(snapshot['constraint_weights'], problem.conWgt)
    if nObj == 1:
        historyCons = np.vstack([p['cons'] for p in result.history.populations])
        expected = calcConstraintViolation(historyCons, problem.conWgt).min()
        assert calcConstraintViolation(result.bestCons, problem.conWgt)[0] == pytest.approx(expected)


def test_weighted_multiobjective_sort_and_selection():
    objs = np.array([[0., 0.], [100., 100.], [50., 50.]])
    cons = np.array([[2., 0.], [0., 3.], [1., 0.]])
    front, last = NDSort(objs, cons, nSort=1, conWgt=[10, 1])
    np.testing.assert_array_equal(front, [3, 1, 2])
    assert last == 1
    pop = Population([[0], [1], [2]], objs, cons, conWgt=[10, 1])
    np.testing.assert_array_equal(pop.argsort(), [1, 2, 0])
    keep, _, _ = NSGAII(**QUIET).environmentalSelection(pop.decs, objs, cons, pop.conWgt, 1)
    np.testing.assert_array_equal(np.flatnonzero(keep), [1])
    nsga = NSGAIII(**QUIET)
    nsga.rng = np.random.default_rng(1)
    keep, _ = nsga.environmentSelection(objs, cons, np.array([[.5, .5]]),
                                                   np.zeros((1, 2)), conWgt=pop.conWgt)
    np.testing.assert_array_equal(np.flatnonzero(keep), [1])
    keep = RVEA(**QUIET).environmentSelection(objs, np.eye(2), .5, cons, pop.conWgt)
    np.testing.assert_array_equal(keep, [1, 2])


def test_zero_weight_is_consistent_for_multiobjective_feasibility():
    pop = Population([[0], [1]], [[0, 0], [1, 1]], [[10, -1], [0, 1]], conWgt=[0, 1])
    best = pop.getBest()
    np.testing.assert_array_equal(best.decs, [[0]])
    problem = problemWithWeights(2, weights=[0, 1])
    algorithm = NSGAII(**QUIET)
    algorithm.setup(problem, 1)
    algorithm.state.update(pop, problem, 2, 0, 'MOEA')
    assert algorithm.state.bestFeasible


def test_sqlite_reader_and_npz_preserve_weights_and_weighted_violation(tmp_path):
    problem = problemWithWeights()
    problem.workDir = str(tmp_path)
    algorithm = GA(nPop=2, maxFEs=2, verboseFlag=False, logFlag=False, saveFlag=True)
    result = algorithm.run(problem, initialPop=[[10, 10], [20, 20]], seed=1)
    np.testing.assert_array_equal(result.bestDecs, [[20, 20]])
    payload = algorithm.saveResult()
    np.testing.assert_array_equal(payload['constraint_weights'], [[10, 1]])
    reader = OptReader(next(tmp_path.rglob('*.sqlite3')))
    try:
        pop = reader.load_last_population()
        best = reader.load_last_best()
        for saved in [pop, best]:
            np.testing.assert_array_equal(saved.conWgt, [[10, 1]])
        np.testing.assert_array_equal(pop.getBest().decs, [[20, 20]])
        assert reader.list_snapshots()[-1]['constraintViolation'] == 3
    finally:
        reader.close()


@pytest.mark.parametrize('kind', ['EGO', 'ASMO', 'MOASMO'])
def test_expensive_optimizers_preserve_weights_after_true_evaluation(kind):
    from UQPyL.optimization.expensive import EGO, ASMO, MOASMO
    from UQPyL.surrogate import MultiSurrogate
    from UQPyL.surrogate.kriging import KRG
    problem = problemWithWeights(2 if kind == 'MOASMO' else 1)
    if kind == 'EGO':
        algorithm = EGO(nInit=6, maxFEs=7, maxIters=1, **QUIET)
        algorithm.optimizer = GA(nPop=4, maxFEs=8, maxIters=1, **QUIET)
    elif kind == 'ASMO':
        algorithm = ASMO(nInit=6, maxFEs=7, maxIters=1,
                         optimizer=GA(nPop=4, maxFEs=8, maxIters=1, **QUIET), **QUIET)
    else:
        algorithm = MOASMO(nInit=6, maxFEs=7, maxIters=1, pct=.2,
                           surrogates=MultiSurrogate(2, models_list=[KRG(), KRG()]),
                           optimizer=NSGAII(nPop=4, maxFEs=8, maxIters=1, **QUIET), **QUIET)
    result = algorithm.run(problem, seed=2)
    assert result.FEs == 7
    np.testing.assert_array_equal(result.extra['constraint_weights'], [[10, 1]])
    for snapshot in result.history.populations:
        np.testing.assert_array_equal(snapshot['constraint_weights'], [[10, 1]])


def test_rvea_feasible_candidates_exclude_infeasible_even_with_better_objectives():
    objs = np.array([[0., 0.], [10., 11.], [11., 10.]])
    cons = np.array([[1., 0.], [0., 0.], [0., 0.]])
    selection = RVEA(**QUIET).environmentSelection(objs, np.eye(2), .5, cons, [10, 1])
    assert len(selection)
    assert 0 not in selection
