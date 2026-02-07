import numpy as np
import pytest

from UQPyL.analysis import Sobol
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _obj(x):
    x = np.asarray(x)
    return float(np.sum(x) + 0.1 * np.sum(x**2))


def test_sobol_sample_skip_value_validation_branches():
    problem = Problem(nInput=3, nOutput=1, ub=1.0, lb=0.0, objFunc=_obj)
    sob = Sobol(verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.raises(ValueError):
        sob.sample(problem, N=8, skipValue=-1)  # negative
    with pytest.raises(ValueError):
        sob.sample(problem, N=8, skipValue=3)  # not power of 2
    with pytest.raises(ValueError):
        sob.sample(problem, N=4, skipValue=8)  # N < skipValue


def test_sobol_sample_skip_value_fast_forward_branch():
    problem = Problem(nInput=3, nOutput=1, ub=1.0, lb=0.0, objFunc=_obj)
    sob = Sobol(verboseFlag=False, logFlag=False, saveFlag=False)

    X = sob.sample(problem, N=8, secondOrder=False, skipValue=4, scramble=False, seed=123)
    assert X.shape == ((problem.nInput + 2) * 8, problem.nInput)


def test_sobol_analyze_divisibility_validation_branches():
    problem = Problem(nInput=3, nOutput=1, ub=1.0, lb=0.0, objFunc=_obj)
    sob = Sobol(verboseFlag=False, logFlag=False, saveFlag=False)

    # secondOrder=True requires divisible by (2*nInput+2)=8
    X_bad = np.zeros((7, 3))
    with pytest.raises(ValueError):
        sob.analyze(problem, X_bad, Y=np.zeros((7, 1)), secondOrder=True)

    # secondOrder=False requires divisible by (nInput+2)=5
    X_bad2 = np.zeros((6, 3))
    with pytest.raises(ValueError):
        sob.analyze(problem, X_bad2, Y=np.zeros((6, 1)), secondOrder=False)


