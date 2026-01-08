import numpy as np

from UQPyL.analysis import Morris
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _sum_obj(x):
    return float(np.sum(x))


def test_morris_sample_and_analyze_smoke():
    problem = Problem(nInput=3, nOutput=1, ub=1.0, lb=0.0, objFunc=_sum_obj, optType="min")
    method = Morris(verboseFlag=False, logFlag=False, saveFlag=False)

    X = method.sample(problem, numTrajectory=5, numLevels=4, seed=123)
    assert X.shape == (5 * (problem.nInput + 1), problem.nInput)

    Y = problem.objFunc(X)
    ds = method.analyze(problem, X, Y=Y, target="objFunc", index="all")
    assert "mu" in ds.data_vars
    assert "mu_star" in ds.data_vars
    assert ds["mu"].shape == (1, problem.nInput)


