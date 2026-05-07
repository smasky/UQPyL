import numpy as np

from UQPyL.analysis import DeltaTest
from UQPyL.doe import LHS
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _sum_obj(x):
    return float(np.sum(x))


def test_delta_test_sample_and_analyze_smoke():
    problem = Problem(nInput=3, nObj=1, ub=1.0, lb=0.0, objFunc=_sum_obj, optType="min")

    method = DeltaTest(nNeighbors=1, verboseFlag=False, logFlag=False, saveFlag=False)

    X = LHS("classic").sample(problem, 30, seed=123)
    assert X.shape == (30, 3)
    assert np.all(X >= problem.lb - 1e-12)
    assert np.all(X <= problem.ub + 1e-12)

    res = method.analyze(problem, X, Y=None, target="objs", index="all")
    metricNames = {metric.name for metric in res.metrics}
    assert "S1" in metricNames
    assert "S1_norm" in metricNames
    s1Metric = next(metric for metric in res.metrics if metric.name == "S1")
    assert s1Metric.values.shape == (1, problem.nInput)


