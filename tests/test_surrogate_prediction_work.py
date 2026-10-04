"""Surrogate prediction work.

Migrated from test_remaining_review.py; original regression provenance is retained below.
"""

import numpy as np
import pytest


# Regression source: test_remaining_review.py::testKernelDiagonalMatchesFullKernel
@pytest.mark.parametrize("kernelName", ["RBF", "Matern", "RationalQuadratic", "Constant", "DotProduct"])
def testKernelDiagonalMatchesFullKernel(kernelName):
    from UQPyL.surrogate.gp import kernel as kernels
    from UQPyL.surrogate.gp.kernel.c_kernel_ import Constant
    from UQPyL.surrogate.gp.kernel.dot_kernel_ import DotProduct

    kernel = ({"Constant": Constant, "DotProduct": DotProduct}.get(kernelName) or getattr(kernels, kernelName))()
    X = np.random.default_rng(4).normal(size=(80, 3))
    np.testing.assert_allclose(kernel.diag(X), np.diag(kernel(X)), rtol=1e-14)


# Regression source: test_remaining_review.py::testMeanPredictionSkipsVarianceWork
@pytest.mark.parametrize("family", ["GPR", "KRG"])
def testMeanPredictionSkipsVarianceWork(family, monkeypatch):
    import importlib
    from UQPyL.surrogate.gp import GPR
    from UQPyL.surrogate.kriging import KRG

    model = {"GPR": GPR, "KRG": KRG}[family]()
    X = np.linspace(0, 1, 12)[:, None]
    model.fitModel(X, np.sin(X))
    mean, variance = model.predict(X, returnVar=True)
    module = importlib.import_module(model.__class__.__module__)

    def unexpected(*args, **kwargs):
        raise AssertionError("Mean prediction must skip variance solves")

    monkeypatch.setattr(module, "solve_triangular" if family == "GPR" else "lstsq", unexpected)
    np.testing.assert_allclose(model.predict(X), mean)
