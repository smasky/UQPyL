import numpy as np
import pytest

from UQPyL.optimization.soea import GA
from UQPyL.optimization.moea import MOEAD
from UQPyL.problem import Problem


@pytest.mark.parametrize('family', [GA, MOEAD])
@pytest.mark.parametrize('budget,expected', [(0, 8), (3, 8), (8, 8), (9, 16), (16, 16), (19, 24)])
def testBudgetCheckedBetweenCompleteIterations(family, budget, expected):
    calls = []

    def objective(x):
        calls.append(len(x))
        y = np.sum(x*x, axis=1)[:, None]
        return np.hstack([y, 2-y]) if family is MOEAD else y

    problem = Problem(nInput=2, nObj=2 if family is MOEAD else 1,
                      lb=0., ub=1., objFunc=objective)
    algorithm = family(nPop=8, maxFEs=budget, maxIters=10, tolerate=None,
                       verboseFlag=False, logFlag=False, saveFlag=False)
    result = algorithm.run(problem, seed=2)
    assert result.FEs == sum(calls) == expected
    assert result.iters == (expected-8)//8
