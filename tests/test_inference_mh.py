import numpy as np
import pytest

from UQPyL.inference import MH
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _quad_obj(x):
    # simple convex objective to make MH acceptance deterministic enough for smoke testing
    x = np.asarray(x)
    return float(np.sum(x**2))


def test_mh_setup_rejects_multi_output_problem():
    problem = Problem(nInput=2, nObj=2, ub=1.0, lb=0.0, objFunc=lambda X: np.zeros((np.atleast_2d(X).shape[0], 2)), optType="min")
    mh = MH(nChains=2, warmUp=0, maxIters=5, verboseFlag=False, logFlag=False, saveFlag=False)
    with pytest.raises(ValueError):
        mh.setup(problem, seed=123)


def test_mh_run_smoke_returns_inf_result():
    problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=_quad_obj, optType="min")
    mh = MH(nChains=2, warmUp=0, maxIters=5, verboseFlag=False, verboseFreq=1000, logFlag=False, saveFlag=False)

    res = mh.run(problem, gamma=0.05, seed=123)
    assert res.decs.shape == (2, 5, 2)
    assert res.objs.shape == (2, 5, 1)
    assert res.logProb.shape == (2, 5)
    assert res.acceptanceRate.shape == (2,)


