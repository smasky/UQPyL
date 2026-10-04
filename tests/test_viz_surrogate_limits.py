"""Viz surrogate limits.

Migrated from test_review_c06_c12.py; original regression provenance is retained below.
"""

import numpy as np
import pytest


# Regression source: test_review_c06_c12.py::testSurrogatePlotIncludesAllData
@pytest.mark.parametrize("values", [[1, 5], [-10, -5], [-2, 4], [-3, -3], [0, 0], [4, 4]])
def testSurrogatePlotIncludesAllData(values, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from UQPyL.viz.surrogate import plot_surrogate

    monkeypatch.setattr(plt, "show", lambda: None)
    yTrue = np.array(values, dtype=float)
    yPred = yTrue * 1.1
    with np.errstate(divide="ignore", invalid="ignore"):
        figure, axes = plot_surrogate("test", yPred, yTrue)
    for limits, data in [(axes.get_xlim(), yTrue), (axes.get_ylim(), yPred)]:
        assert limits[0] < data.min() <= data.max() < limits[1]
    plt.close(figure)


# Regression source: test_review_c06_c12.py::testSurrogatePlotHonorsExplicitLimits
def testSurrogatePlotHonorsExplicitLimits(monkeypatch):
    import matplotlib.pyplot as plt
    from UQPyL.viz.surrogate import plot_surrogate

    monkeypatch.setattr(plt, "show", lambda: None)
    figure, axes = plot_surrogate("test", [-10, -5], [-10, -5], ylim=(-8, -6))
    assert axes.get_xlim() == axes.get_ylim() == (-8, -6)
    plt.close(figure)
