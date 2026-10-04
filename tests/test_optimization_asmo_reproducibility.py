"""Optimization asmo reproducibility.

Migrated from test_review_a07_a15.py; original regression provenance is retained below.
"""

import numpy as np
from UQPyL.problem import Problem, Sphere
from UQPyL.surrogate.kriging import KRG
from UQPyL.optimization.expensive import ASMO
from UQPyL.optimization.soea import GA

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


# Regression source: test_review_a07_a15.py::testAsmoSeedControlsActualKrgFitsAndEvaluations
def testAsmoSeedControlsActualKrgFitsAndEvaluations():
    records = []
    for _ in range(2):
        evaluated = []

        def objective(x):
            evaluated.append(x.copy())
            return np.sin(5 * x) + x * x

        problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=objective)
        alg = ASMO(
            nInit=8,
            maxFEs=10,
            maxIters=2,
            surrogate=KRG(),
            optimizer=GA(nPop=8, maxFEs=16, maxIters=1, **QUIET),
            **QUIET,
        )
        result = alg.run(problem, seed=12)
        records.append((np.vstack(evaluated), result.bestObjs, alg.surrogate.predict(np.array([[0.25], [0.75]]))))
    for a, b in zip(*records):
        np.testing.assert_array_equal(a, b)
