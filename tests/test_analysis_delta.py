import numpy as np

from UQPyL.analysis import DeltaTest
from UQPyL.doe import LHS
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _sum_obj(x):
    return float(np.sum(x))


def test_delta_test_sample_and_analyze_smoke():
    problem = Problem(nInput=3, nOutput=1, ub=1.0, lb=0.0, objFunc=_sum_obj, optType="min")

    method = DeltaTest(verboseFlag=False, logFlag=False, saveFlag=False)

    X = method.sample(problem, N=30, sampler=LHS("classic"), seed=123)
    assert X.shape == (30, 3)
    assert np.all(X >= problem.lb - 1e-12)
    assert np.all(X <= problem.ub + 1e-12)

    ds = method.analyze(problem, X, Y=None, target="objFunc", index="all", nNeighbors=1)
    # Should be an xarray.Dataset with expected variables.
    assert "S1" in ds.data_vars
    assert "S1_scale" in ds.data_vars
    assert ds["S1"].shape == (1, problem.nInput)


