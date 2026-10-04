"""Optimization reference point validation.

Migrated from test_review_a01_a06.py; original regression provenance is retained below.
"""

import subprocess
import sys
import pytest
from UQPyL.optimization.core.uniform_point import uniformPoint
from UQPyL.optimization.moea import MOEAD, NSGAIII, RVEA
from UQPyL.problem import Problem


# Regression source: test_review_a01_a06.py::testSingleObjectiveReferencePointTerminates
def testSingleObjectiveReferencePointTerminates():
    code = "from UQPyL.optimization.core.uniform_point import uniformPoint; w,n=uniformPoint(8,1); assert n==1 and w.tolist()==[[1.0]]"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


# Regression source: test_review_a01_a06.py::testReferencePointInputValidation
@pytest.mark.parametrize("method", ["NBI", "grid"])
@pytest.mark.parametrize("n,m", [(0, 2), (8, 0), (-1, 2), (True, 2), (8, False), (2.5, 2), (8, 1.5)])
def testReferencePointInputValidation(method, n, m):
    with pytest.raises(ValueError, match="positive integer"):
        uniformPoint(n, m, method)


# Regression source: test_review_a01_a06.py::testMoeaRejectsSingleObjectiveBeforeEvaluation
@pytest.mark.parametrize("algorithmClass", [MOEAD, NSGAIII, RVEA])
def testMoeaRejectsSingleObjectiveBeforeEvaluation(algorithmClass):
    calls = []
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: calls.append(x))
    with pytest.raises(ValueError, match="at least two objectives"):
        algorithmClass(nPop=8, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, seed=1)
    assert not calls
