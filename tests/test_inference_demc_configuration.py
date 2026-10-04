"""Inference demc configuration.

Migrated from test_review_a07_a15.py; original regression provenance is retained below.
"""

import pytest
from UQPyL.problem import Problem, Sphere
from UQPyL.inference import DEMC

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


# Regression source: test_review_a07_a15.py::testDemcInvalidChainsAtConstruction
@pytest.mark.parametrize("count", [1, 2, True, 3.5])
def testDemcInvalidChainsAtConstruction(count):
    with pytest.raises(ValueError, match="nChains"):
        DEMC(nChains=count, **QUIET)


# Regression source: test_review_a07_a15.py::testDemcDefaultRuns
def testDemcDefaultRuns():
    model = DEMC(warmUp=0, maxIters=2, **QUIET)
    model.run(Sphere(nInput=1), seed=3)
    assert model.get("nChains") == 3
