"""Optimization ego component contracts.

Migrated from test_review_remaining.py; original regression provenance is retained below.
"""

import pytest
from UQPyL.optimization.expensive import EGO
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem, ModelProblem
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.rbf import RBF


# Regression source: test_review_remaining.py::testEgoComponentInjectionAndDefaultIsolation
def testEgoComponentInjectionAndDefaultIsolation():
    surrogate = KRG(nRestartTimes=0)
    optimizer = GA(nPop=4, maxIters=0, verboseFlag=False)
    method = EGO(nInit=4, maxIters=1, surrogate=surrogate, optimizer=optimizer, verboseFlag=False)
    assert method.surrogate is surrogate and method.optimizer is optimizer
    problem = Problem(nInput=1, nObj=1, lb=-1, ub=1, objFunc=lambda x: x**2)
    assert method.run(problem, seed=2).FEs == 5
    first, second = EGO(), EGO()
    assert first.surrogate is not second.surrogate and first.optimizer is not second.optimizer
    calls = []
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: calls.append(x) or x**2)
    with pytest.raises(ValueError, match="variance"):
        EGO(surrogate=RBF()).run(problem, seed=2)
    assert calls == []
