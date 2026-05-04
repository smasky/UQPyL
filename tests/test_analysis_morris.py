import numpy as np
import pytest

from UQPyL.doe import MorrisDesign
from UQPyL.analysis import Morris
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _sum_obj(x):
    return float(np.sum(x))


def test_morris_sample_and_analyze_smoke():
    problem = Problem(nInput=3, nObj=1, ub=1.0, lb=0.0, objFunc=_sum_obj, optType="min")
    method = Morris(verboseFlag=False, logFlag=False, saveFlag=False)

    X, meta = MorrisDesign(numLevels=4).sampleWithMeta(problem, 5, seed=123)
    assert X.shape == (5 * (problem.nInput + 1), problem.nInput)

    Y = problem.objFunc(X)
    res = method.analyze(problem, X, Y=Y, meta=meta, target="objs", index="all")
    metricNames = {metric.name for metric in res.metrics}
    assert "mu" in metricNames
    assert "mu_star" in metricNames
    assert res.meta["designType"] == "morris"
    muMetric = next(metric for metric in res.metrics if metric.name == "mu")
    assert muMetric.values.shape == (1, problem.nInput)


def test_morris_invalid_trajectory_structure_raises():
    problem = Problem(nInput=3, nObj=1, ub=1.0, lb=0.0, objFunc=_sum_obj, optType="min")
    method = Morris(verboseFlag=False, logFlag=False, saveFlag=False)

    X = np.zeros((4, 3))
    Y = np.zeros((4, 1))
    meta = {"designType": "morris", "numLevels": 4}

    with pytest.raises(ValueError):
        method.analyze(problem, X, Y=Y, meta=meta, target="objs", index="all")


