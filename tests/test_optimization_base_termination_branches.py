import numpy as np

from UQPyL.optimization.base import AlgorithmABC
from UQPyL.optimization.population import Population
from UQPyL.problem import ModelProblem, Problem


class _Emit:
    def send(self):
        return None

    def unfinished(self):
        return None

    def finished(self):
        return None


class _GuiProblem(Problem):
    name = "GuiProblem"

    def __init__(self):
        super().__init__(nInput=1, nObj=1, ub=1.0, lb=0.0, nCon=0, optType="min")
        self.GUI = True
        self.totalWidth = 110
        self.iterEmit = _Emit()
        self.isStop = False

    def objFunc(self, X):
        X = self._check_X_2d(X)
        return X.copy()


class _Alg(AlgorithmABC):
    name = "Alg"
    alg_type = "EA"

    def run(self, problem, seed=None):
        raise NotImplementedError


def test_checktermination_gui_stop_preserves_counters():
    problem = _GuiProblem()
    alg = _Alg(maxFEs=10, maxIters=10, tolerate=1e-6, maxTolerates=10, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=123)

    pop = Population(np.array([[0.1]]), np.array([[0.1]]))
    alg.update(pop)

    # GUI stop branch
    problem.isStop = True
    assert alg.checkTermination(pop) is False
    assert alg.iters == alg.tolerateTimes == 0

    # Resuming without a completed iteration does not consume the allowance.
    problem.isStop = False
    alg.FEs = 0
    alg.iters = 0
    alg.tolerateTimes = 0
    assert alg.checkTermination(pop) is True
    assert alg.iters == alg.tolerateTimes == 0


def test_algorithm_evaluate_accepts_model_problem():
    def simf(X):
        X = np.atleast_2d(X)
        return np.sum(X, axis=1)

    def objf(X, context):
        return context.sims.reshape(-1, 1)

    problem = ModelProblem(nInput=2, nObj=1, ub=1.0, lb=0.0, simFunc=simf, objFunc=objf)
    alg = _Alg(maxFEs=10, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setProblem(problem)
    alg.FEs = 0
    pop = Population(np.array([[0.1, 0.2], [0.3, 0.4]]))

    alg.evaluate(pop)

    assert np.allclose(pop.objs, [[0.3], [0.7]])
    assert pop.cons is None
    assert alg.FEs == 2


def test_initpop_accepts_decision_matrix_and_fills_remaining():
    problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=lambda X: np.sum(np.atleast_2d(X) ** 2, axis=1, keepdims=True), optType="min")
    alg = _Alg(maxFEs=10, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=123)

    initial = np.array([[0.0, 0.0], [0.5, 0.0]])
    pop = alg.initPop(4, initialPop=initial)

    assert len(pop) == 4
    assert np.allclose(pop.decs[:2], problem.space_to_unit(initial))
    assert np.allclose(problem.unit_to_space(pop.decs[:2]), initial)
    assert pop.isEvaluated
    assert alg.FEs == 4


def test_initpop_uses_evaluated_population_without_reevaluation():
    problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=lambda X: np.sum(np.atleast_2d(X) ** 2, axis=1, keepdims=True), optType="min")
    alg = _Alg(maxFEs=10, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=123)

    initial = Population(
        np.array([[0.0, 0.0], [0.5, 0.0]]),
        objs=np.array([[0.0], [0.25]]),
    )
    pop = alg.initPop(2, initialPop=initial)

    assert len(pop) == 2
    assert np.allclose(pop.objs, initial.objs)
    assert alg.FEs == 0


def test_initpop_rejects_oversized_initial_population():
    problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=lambda X: np.sum(np.atleast_2d(X) ** 2, axis=1, keepdims=True), optType="min")
    alg = _Alg(maxFEs=10, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=123)

    initial = np.zeros((3, 2))

    try:
        alg.initPop(2, initialPop=initial)
    except ValueError as err:
        assert "initialPop has 3 members" in str(err)
    else:
        raise AssertionError("Expected oversized initialPop to raise ValueError.")
