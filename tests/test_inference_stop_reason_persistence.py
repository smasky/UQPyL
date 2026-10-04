"""Inference stop reason persistence.

Migrated from test_review_c16_c18_c19_c21.py; original regression provenance is retained below.
"""

from algorithm_capability_test_support import makeProblem
import numpy as np
import pytest
from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS, InfReader
from UQPyL.problem import Problem


# Regression source: test_review_c16_c18_c19_c21.py::testInferenceStopReasonRoundTrip
@pytest.mark.parametrize("methodClass", [MH, MH_Gibbs, AMH, DEMC, DREAM_ZS])
def testInferenceStopReasonRoundTrip(methodClass, tmp_path):
    options = {"maxIters": 3}
    method = methodClass(nChains=4, warmUp=0, saveFlag=True, saveFreq=1, logFlag=False, verboseFlag=False, **options)
    problem = makeProblem()
    problem.workDir = str(tmp_path)
    result = method.run(problem, seed=1)
    assert result.stopReason == result.summary()["stop_reason"] == "max_iters"
    assert result.history.snapshots[-1]["stop_reason"] == "max_iters"
    with InfReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        assert reader.get_run_summary()["stop_reason"] == reader.load_result().stopReason == "max_iters"
    method.reset()
    assert result.stopReason == "max_iters" and method.state.stopReason is None
