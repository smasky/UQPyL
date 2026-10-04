"""Calibration gain work.

Migrated from test_review_remaining.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from UQPyL.calibration import IES
from UQPyL.calibration.methods._ensemble import ensembleGain
from UQPyL.problem import Problem, ModelProblem


# Regression source: test_review_remaining.py::testSingleDecompositionGainMatchesIndependentReference
@pytest.mark.parametrize("nObs,rank,lam", [(3, 3, 0), (20, 3, 0), (20, 3, 0.2), (5, 0, 0)])
def testSingleDecompositionGainMatchesIndependentReference(nObs, rank, lam, monkeypatch):
    rng = np.random.default_rng(12)
    y = rng.normal(size=(rank, nObs))
    cxy = rng.normal(size=(2, rank)) @ y
    cyy = y.T @ y
    r = np.zeros((nObs, nObs))
    matrix = cyy + lam * np.eye(nObs)
    reference = cxy @ np.linalg.pinv(matrix, rcond=nObs * np.finfo(float).eps)
    originals = [a.copy() for a in (cxy, cyy, r)]
    eigh = np.linalg.eigh
    calls = []

    def counted(a):
        calls.append(a.shape)
        return eigh(a)

    monkeypatch.setattr(np.linalg, "eigh", counted)
    monkeypatch.setattr(np.linalg, "solve", lambda *a: pytest.fail("Redundant decomposition"))
    gain, info = ensembleGain(cxy, cyy, r, lam)
    np.testing.assert_allclose(gain, reference, atol=2e-12, rtol=2e-11)
    assert calls == [(nObs, nObs)]
    assert info["rank"] == (nObs if lam else rank)
    for original, current in zip(originals, (cxy, cyy, r)):
        np.testing.assert_array_equal(current, original)


# Regression source: test_review_remaining.py::testFixedNoiseValidatedOnceAcrossIesIterations
def testFixedNoiseValidatedOnceAcrossIesIterations(monkeypatch):
    sizes = []
    original = np.linalg.eigh

    def counted(a):
        sizes.append(a.shape)
        return original(a)

    monkeypatch.setattr(np.linalg, "eigh", counted)
    problem = ModelProblem(
        nInput=1, lb=-10, ub=10, obs=np.ones(5), simFunc=lambda x: (np.repeat(x[:, None, :], 5, axis=1)).reshape(len(x), -1)
    )
    IES(maxIters=3).run(problem, np.array([[-1.0], [0.0], [2.0]]), r=np.eye(5))
    assert sizes == [(5, 5)] * 5  # Validation, one noise square root, and one gain per round.
