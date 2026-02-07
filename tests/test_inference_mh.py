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
    problem = Problem(nInput=2, nOutput=2, ub=1.0, lb=0.0, objFunc=_quad_obj, optType="min")
    mh = MH(nChains=2, warmUp=0, maxIters=5, verboseFlag=False, logFlag=False, saveFlag=False)
    with pytest.raises(ValueError):
        mh.setup(problem, seed=123)


def test_mh_run_smoke_returns_expected_datasets():
    problem = Problem(nInput=2, nOutput=1, ub=1.0, lb=-1.0, objFunc=_quad_obj, optType="min")
    mh = MH(nChains=2, warmUp=0, maxIters=5, verboseFlag=False, verboseFreq=1000, logFlag=False, saveFlag=False)

    res = mh.run(problem, gamma=0.05, seed=123)
    assert "posterior" in res
    assert "stats" in res
    assert "optimization" in res

    posterior = res["posterior"]
    assert "decs" in posterior.data_vars
    assert "objs" in posterior.data_vars

    stats = res["stats"]
    assert "acceptanceRate_mean" in stats.data_vars


