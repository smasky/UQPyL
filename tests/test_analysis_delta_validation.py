"""Analysis delta validation.

Migrated from test_review_a07_a15.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from UQPyL.analysis import DeltaTest
from UQPyL.problem import Problem, Sphere

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


# Regression source: test_review_a07_a15.py::testDeltaRejectsInvalidNeighbors
@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def testDeltaRejectsInvalidNeighbors(count):
    with pytest.raises(ValueError, match="positive integer"):
        DeltaTest(nNeighbors=count)


# Regression source: test_review_a07_a15.py::testDeltaReportsUnsupportedData
@pytest.mark.parametrize("nSamples,nInput", [(8, 1), (2, 2)])
def testDeltaReportsUnsupportedData(nSamples, nInput):
    x = np.arange(nSamples * nInput, dtype=float).reshape(nSamples, nInput)
    with pytest.raises(ValueError, match="at least two inputs|smaller than the sample count"):
        DeltaTest(**QUIET).analyze(Sphere(nInput=nInput), x, np.sum(x * x, axis=1)[:, None])
