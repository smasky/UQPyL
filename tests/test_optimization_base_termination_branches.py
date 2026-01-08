import numpy as np

from UQPyL.optimization.base import AlgorithmABC
from UQPyL.optimization.population import Population
from UQPyL.problem import ProblemABC


class _Emit:
    def send(self):
        return None

    def unfinished(self):
        return None

    def finished(self):
        return None


class _GuiProblem(ProblemABC):
    name = "GuiProblem"

    def __init__(self):
        super().__init__(nInput=1, nOutput=1, ub=1.0, lb=0.0, nCons=0, optType="min")
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


def test_checktermination_gui_stop_branch_and_tolerate_branch():
    problem = _GuiProblem()
    alg = _Alg(maxFEs=10, maxIters=10, tolerate=1e-6, maxTolerates=10, verboseFlag=False, logFlag=False, saveFlag=False)
    alg.setup(problem, seed=123)

    pop = Population(np.array([[0.1]]), np.array([[0.1]]))
    # ensure tolerate branch is active
    alg.result.bestObjs = np.array([[0.1]])

    # GUI stop branch
    problem.isStop = True
    assert alg.checkTermination(pop) is False

    # tolerateTimes increase branch (no improvement)
    problem.isStop = False
    alg.FEs = 0
    alg.iters = 0
    alg.tolerateTimes = 0
    assert alg.checkTermination(pop) is True
    assert alg.tolerateTimes >= 0

