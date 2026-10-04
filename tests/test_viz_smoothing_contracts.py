"""Viz smoothing contracts.

Migrated from test_review_a07_a15.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from UQPyL.viz.common import smooth_curve


# Regression source: test_review_a07_a15.py::testSmoothingPreservesLengthAndInputs
@pytest.mark.parametrize("length", [0, 1, 3, 9, 10, 40, 60])
def testSmoothingPreservesLengthAndInputs(length):
    x = np.arange(length, dtype=float) ** 2
    original = x.copy()
    result = smooth_curve(x)
    assert result.shape == x.shape
    np.testing.assert_array_equal(x, original)
    if length < 10:
        np.testing.assert_array_equal(result, x)
    if length == 60:
        assert np.any(result[20:-20] != x[20:-20])
