"""Analysis shape contracts.

Migrated from test_remaining_review.py; original regression provenance is retained below.
"""

import numpy as np
import pytest


# Regression source: test_remaining_review.py::testAnalysisShapesAndSelectedLabels
@pytest.mark.parametrize("methodName", ["Sobol", "FAST", "RBDFAST", "RSA", "DeltaTest", "Morris"])
def testAnalysisShapesAndSelectedLabels(methodName):
    import UQPyL.analysis as analysis
    from UQPyL.doe import SaltelliDesign, FASTDesign, MorrisDesign, LHS
    from UQPyL.problem import Problem

    problem = Problem(nInput=2, nObj=2, lb=0.0, ub=1.0, objLabels=["flow", "temperature"], objFunc=lambda X: X)
    design, count = {"Sobol": (SaltelliDesign(), 64), "FAST": (FASTDesign(), 129), "Morris": (MorrisDesign(), 10)}.get(
        methodName, (LHS(), 80)
    )
    X, meta = design.sampleWithMeta(problem, count, seed=1)
    method = getattr(analysis, methodName)(verboseFlag=False)
    options = dict(meta=meta) if methodName in ("Sobol", "FAST", "Morris") else {}
    selected = method.analyze(problem, X, Y=X, index=1, **options)
    single = method.analyze(problem, X, Y=X[:, 1], **options)
    for left, right in zip(selected.metrics, single.metrics):
        assert left.rowLabels == ["temperature"]
        np.testing.assert_allclose(left.values, right.values)
    with pytest.raises(ValueError, match="row"):
        method.analyze(problem, X, Y=X[:-1], **options)
