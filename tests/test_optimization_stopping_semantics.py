import numpy as np
import pytest

from UQPyL.optimization.base import AlgorithmABC
from UQPyL.optimization.population import Population
from UQPyL.optimization.soea import GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.optimization.expensive import EGO, ASMO, MOASMO
from UQPyL.optimization.runtime import OptReader
from UQPyL.problem import Problem
from UQPyL.problem.sop.single_simple_problem import Sphere

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)
METHODS = [GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA, NSGAII, NSGAIII, MOEAD, RVEA, EGO, ASMO, MOASMO]
MULTI = [NSGAII, NSGAIII, MOEAD, RVEA, MOASMO]


def makeMethod(cls, limit):
    options = dict(maxIters=limit, maxFEs=1000, historyFreq=1, **QUIET)
    if cls in [SCE_UA, ML_SCE_UA]:
        return cls(ngs=2, tolerate=None, **options)
    if cls in [EGO, ASMO]:
        method = cls(nInit=6, **options)
        method.optimizer = GA(nPop=6, maxFEs=12, tolerate=None, **QUIET)
        return method
    if cls is MOASMO:
        return cls(nInit=8, pct=.25, optimizer=NSGAII(nPop=8, maxFEs=16, **QUIET), **options)
    return cls(nPop=12, tolerate=None, **options)


@pytest.mark.parametrize("methodClass", METHODS)
@pytest.mark.parametrize("limit", [0, 1, 2])
def test_every_algorithm_counts_completed_iterations_exactly(methodClass, limit):
    multi = methodClass in MULTI
    problem = Problem(nInput=2, nObj=2 if multi else 1, lb=-5, ub=5,
                      objFunc=lambda X: np.column_stack([np.sum(X**2, axis=1), np.sum((X-1)**2, axis=1)])[:, :2 if multi else 1])
    method = makeMethod(methodClass, limit)
    result = method.run(problem, seed=2)
    assert method.iters == result.iters == limit
    assert [pair[0] for pair in result.history.iterToFEs] == list(range(limit + 1))
    assert result.history.snapshotIterToFEs == result.history.iterToFEs
    assert result.history.iterToFEs[-1] == [result.iters, result.FEs]
    assert 0 <= result.appearIters <= result.iters


class ScriptedAlgorithm(AlgorithmABC):
    name = "Scripted"
    alg_type = "EA"

    def run(self, problem, rows, seed=1):
        self.setup(problem, seed)
        rows = iter(rows)

        def population(row):
            objective, cons = row
            self.FEs += 1
            return Population([[.5]], [[objective * problem.opt]],
                              None if cons is None else [cons], problem.conWgt)

        pop = population(next(rows))
        self.update(pop)
        while self.checkTermination(pop):
            try:
                pop = population(next(rows))
            except StopIteration:
                break
            self.update(pop, completed=True)
        return self.finalize()


def scripted(rows, *, tolerate=0., maxTolerates=2, optType="min", weights=None):
    nCon = 0 if weights is None else len(weights)
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, optType=optType, nCon=nCon, conWgt=weights,
                      objFunc=lambda X: np.zeros((len(X), 1)),
                      conFunc=None if nCon == 0 else lambda X: np.zeros((len(X), nCon)))
    method = ScriptedAlgorithm(maxFEs=100, maxIters=20, maxTolerates=maxTolerates,
                               tolerate=tolerate, historyFreq=1, **QUIET)
    return method, method.run(problem, rows)


@pytest.mark.parametrize("direction", ["min", "max"])
def test_stagnation_starts_after_initialization_and_resets_on_improvement(direction):
    sign = 1 if direction == "min" else -1
    method, result = scripted([(sign*v, None) for v in [10, 10, 8, 8, 7, 7, 7, 6]], optType=direction)
    assert result.iters == 6 and method.tolerateTimes == 2
    assert result.bestObjs.item() == sign*7


@pytest.mark.parametrize("values,tolerance,expectedIters", [
    ([10, 9.5, 9, 8], 1., 2),  # Small improvements still count as stagnation.
    ([10, 9, 8, 7], 1., 2),    # Equality with the threshold is insufficient.
    ([10, 10, 20, 5], 0., 2),  # A worse current population is not progress.
    ([10, 8, 6, 6, 6], 1., 4),
])
def test_objective_tolerance_uses_new_historical_best(values, tolerance, expectedIters):
    method, result = scripted([(v, None) for v in values], tolerate=tolerance)
    assert result.iters == expectedIters and method.tolerateTimes == 2


def test_weighted_constraint_progress_resets_stagnation_even_when_objective_worsens():
    rows = [(0, [2, 0]), (0, [2, 0]), (100, [1, 100]),
            (100, [1, 100]), (200, [0, 100]), (200, [0, 100]), (200, [0, 100]), (-10, [0, 0])]
    method, result = scripted(rows, tolerate=1., weights=[1, 0])
    assert result.iters == 6 and method.tolerateTimes == 2
    assert result.bestObjs.item() == 200 and result.bestFeasible


@pytest.mark.parametrize("rows", [
    [(10, [1]), (5, [1]), (0, [1]), (-5, [1])],  # Equal violation: better objective is irrelevant.
    [(10, [0]), (5, [1]), (0, [2]), (-5, [0])],  # Losing feasibility is not progress.
])
def test_infeasible_objective_changes_do_not_reset_stagnation(rows):
    method, result = scripted(rows, weights=[1])
    assert result.iters == 2 and method.tolerateTimes == 2
    assert result.bestObjs.item() == 10


@pytest.mark.parametrize("tolerate,maxTolerates", [(None, 0), (None, 2), (0., None)])
def test_disabled_stagnation_does_not_stop_flat_sequence(tolerate, maxTolerates):
    _, result = scripted([(1, None)] * 6, tolerate=tolerate, maxTolerates=maxTolerates)
    assert result.iters == 5


def test_zero_stagnation_allowance_stops_after_initialization():
    method, result = scripted([(1, None)] * 3, maxTolerates=0)
    assert result.iters == 0 and method.tolerateTimes == 0


def test_repeated_termination_checks_do_not_change_counters_or_history():
    method = GA(nPop=4, maxFEs=100, maxTolerates=2, **QUIET)
    method.setup(Sphere(nInput=1), seed=1)
    pop = method.initPop(4)
    method.update(pop)
    for _ in range(3):
        assert method.checkTermination(pop)
    assert method.iters == method.tolerateTimes == 0
    assert len(method.state.history.iterToFEs) == 1
    method.maxIter = 0
    for _ in range(3):
        assert not method.checkTermination(pop)
    assert method.iters == method.tolerateTimes == 0


def test_real_ga_does_not_stop_during_continuous_improvement():
    problem = Sphere(nInput=5, lb=-5, ub=5)
    options = dict(nPop=40, maxFEs=4000, tolerate=1e-12, historyFreq=1, **QUIET)
    control = GA(maxTolerates=None, **options).run(problem, seed=2)
    best = np.array(control.history.bestObjHistory)
    stagnant = 0
    expected = len(best)-1
    for iteration in range(1, len(best)):
        stagnant = 0 if best[iteration-1] - best[iteration] > 1e-12 else stagnant+1
        if stagnant == 2:
            expected = iteration
            break
    method = GA(maxTolerates=2, **options)
    result = method.run(problem, seed=2)
    assert result.FEs > 160
    assert result.iters == expected
    np.testing.assert_array_equal(result.history.bestObjHistory, best[:expected+1])


@pytest.mark.parametrize("budget,expectedIters,expectedFEs", [(40, 0, 40), (41, 1, 80)])
def test_existing_batch_budget_rule_is_preserved(budget, expectedIters, expectedFEs):
    result = GA(nPop=40, maxFEs=budget, tolerate=None, **QUIET).run(Sphere(nInput=2), seed=2)
    assert result.iters == expectedIters and result.FEs == expectedFEs


def test_exhausted_discrete_search_does_not_count_an_unfinished_iteration():
    problem = Problem(nInput=1, nObj=1, lb=-100, ub=100, varType=[2], varSet={0: [10, 20]}, objFunc=lambda X: X**2)
    method = EGO(nInit=2, maxFEs=10, maxIters=4, **QUIET)
    method.optimizer = GA(nPop=4, maxFEs=8, **QUIET)
    result = method.run(problem, initialPop=[[10], [20]], seed=3)
    assert result.iters == 0 and result.FEs == 2


def test_multiobjective_archive_discovery_uses_completed_iteration_and_exact_evaluation():
    problem = Problem(nInput=1, nObj=2, lb=0, ub=1,
                      objFunc=lambda X: np.repeat(X, 2, axis=1))
    method = NSGAII(nPop=4, maxFEs=100, maxTolerates=0, **QUIET)
    method.setup(problem, seed=2)
    initial = method.evaluate(Population([[.8]]))
    method.update(initial)
    assert method.state.appearIters == 0
    assert method.checkTermination(initial)  # Single-objective stagnation is inapplicable.
    method.evaluate(Population([[.2]]))
    selected = method.evaluate(Population([[.6]]))
    method.update(selected, completed=True)
    assert method.state.appearIters == 1 and method.state.appearFEs == 2
    np.testing.assert_array_equal(method.state.bestObjs, [[.2, .2]])
    method.update(method.evaluate(selected), completed=True)
    assert method.state.appearIters == 1 and method.state.appearFEs == 2
    assert method.iters == 2 and method.tolerateTimes == 0


def test_asmo_one_step_counts_one_completed_iteration():
    method = makeMethod(ASMO, 10)
    result = method.run(Sphere(nInput=2), seed=2, oneStep=True)
    assert result.iters == 1
    assert [pair[0] for pair in result.history.iterToFEs] == [0, 1]


def test_reusing_algorithm_resets_iteration_and_stagnation_counts():
    method = GA(nPop=8, maxFEs=1000, maxTolerates=2, historyFreq=1, **QUIET)
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1,
                      objFunc=lambda X: np.ones((len(X), 1)))
    first = method.run(problem, seed=2)
    second = method.run(problem, seed=2)
    assert first.iters == second.iters == 2 and method.tolerateTimes == 2
    assert first.FEs == second.FEs == 24
    assert first.history.iterToFEs == second.history.iterToFEs
    np.testing.assert_array_equal(first.bestDecs, second.bestDecs)


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_saved_iteration_numbers_match_completed_history(tmp_path, limit):
    problem = Sphere(nInput=2)
    problem.workDir = str(tmp_path)
    method = GA(nPop=8, maxFEs=1000, maxIters=limit, tolerate=None,
                saveFlag=True, saveFreq=1, historyFreq=1, verboseFlag=False, logFlag=False)
    result = method.run(problem, seed=2)
    reader = OptReader(next(tmp_path.rglob('*.sqlite3')))
    try:
        assert reader.get_run_summary()['final_iters'] == limit
        assert [s['iter'] for s in reader.list_snapshots()] == list(range(limit+1)) + [limit]
    finally:
        reader.close()
    assert result.iters == limit
