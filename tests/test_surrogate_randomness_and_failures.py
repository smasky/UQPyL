"""Surrogate randomness and failures.

Migrated from test_review_a07_a15.py; original regression provenance is retained below.
"""

import builtins
import importlib
import numpy as np
import pytest
from scipy.stats import kendalltau
from UQPyL.surrogate.metric import rank_score
from UQPyL.surrogate import MultiSurrogate
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.regression.linear_regression import LinearRegression


# Regression source: test_review_a07_a15.py::testRankUsesEveryOutputAndHandlesTies
def testRankUsesEveryOutputAndHandlesTies():
    actual = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    predicted = np.array([[0.0, 30.0], [1.0, 20.0], [2.0, 10.0]])
    assert rank_score(actual, predicted) == 0.0
    a = np.array([0.0, 0.0, 1.0, 2.0])
    b = np.array([0.0, 1.0, 1.0, 2.0])
    assert rank_score(a, b) == pytest.approx(kendalltau(a, b, variant="b").statistic)
    assert rank_score(np.ones(4), np.arange(4.0)) == 0.0
    with pytest.raises(ValueError, match="two samples"):
        rank_score(np.ones(1), np.ones(1))


# Regression source: test_review_a07_a15.py::testMultiSurrogatePropagatesIndependentReproducibleStreams
def testMultiSurrogatePropagatesIndependentReproducibleStreams():
    x = np.linspace(0, 1, 10)[:, None]
    y = np.hstack([np.sin(4 * x), np.cos(4 * x)])
    traces = []
    for _ in range(2):
        model = MultiSurrogate(2, [KRG(), KRG()])
        model.rng = np.random.default_rng(72)
        model.fit(x, y)
        traces.append((model.predict(x), [m.rng.integers(0, 2**32) for m in model.models_list]))
    assert traces[0][1] == traces[1][1]
    assert traces[0][1][0] != traces[0][1][1]
    # RNG streams are exact; ill-conditioned KRG solves need numerical tolerance.
    np.testing.assert_allclose(traces[0][0], traces[1][0], rtol=1e-8, atol=1e-8)
    for prediction, _ in traces:
        np.testing.assert_allclose(prediction, y, rtol=0, atol=1e-6)


# Regression source: test_review_a07_a15.py::testTunerProgrammingErrorsPropagate
@pytest.mark.parametrize("entry", ["gridTune", "optTune"])
@pytest.mark.parametrize("error", [TypeError("bug"), AttributeError("bug"), ValueError("bad shape")])
def testTunerProgrammingErrorsPropagate(monkeypatch, entry, error):
    model = LinearRegression(lossType="Ridge")

    def fail(*args):
        raise error

    monkeypatch.setattr(model, "fitModel", fail)

    class Optimizer:
        def run(self, problem, seed):
            problem.evaluate(np.array([[0.1]]))

    tuner = AutoTuner(model, Optimizer())
    kwargs = {"paraGrid": {"C": [0.1]}} if entry == "gridTune" else {"paraList": ["C"]}
    x = np.arange(10.0)[:, None]
    with pytest.raises(type(error), match=str(error)):
        getattr(tuner, entry)(x, x * x, ratio=30, tuneMode="joint", seed=1, **kwargs)


# Regression source: test_review_a07_a15.py::testTunerRecordsNumericalFailuresWithoutPrinting
def testTunerRecordsNumericalFailuresWithoutPrinting(monkeypatch, capsys):
    model = LinearRegression(lossType="Ridge")
    original = model.fitModel
    calls = []

    def fit(x, y):
        calls.append(1)
        if len(calls) == 1:
            raise np.linalg.LinAlgError("singular candidate")
        return original(x, y)

    monkeypatch.setattr(model, "fitModel", fit)
    tuner = AutoTuner(model)
    x = np.arange(10.0)[:, None]
    _, score = tuner.gridTune(x, x * x, {"C": [0.1, 0.2]}, ratio=30, seed=1, tuneMode="joint")
    assert np.isfinite(score)
    assert tuner.candidateFailures == [
        {"candidate_index": 0, "error_type": "LinAlgError", "message": "singular candidate"}
    ]
    assert not capsys.readouterr().out


# Regression source: test_review_a07_a15.py::testOptionalMarsDoesNotHideUnexpectedImportFailures
@pytest.mark.parametrize(
    "failure", [RuntimeError("implementation error"), ModuleNotFoundError("unrelated", name="unrelated_dependency")]
)
def testOptionalMarsDoesNotHideUnexpectedImportFailures(monkeypatch, failure):
    module = importlib.import_module("UQPyL.analysis.methods")
    original = builtins.__import__

    def fail(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mars" and globals and globals.get("__package__") == "UQPyL.analysis.methods":
            raise failure
        return original(name, globals, locals, fromlist, level)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", fail)
            with pytest.raises(type(failure), match=str(failure)):
                importlib.reload(module)
    finally:
        importlib.reload(module)


# Regression source: test_review_a07_a15.py::testOptionalMarsMissingExtensionIsRecognized
def testOptionalMarsMissingExtensionIsRecognized(monkeypatch):
    module = importlib.import_module("UQPyL.analysis.methods")
    original = builtins.__import__

    def fail(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mars" and globals and globals.get("__package__") == "UQPyL.analysis.methods":
            raise ModuleNotFoundError("missing extension", name="UQPyL.surrogate.mars.core._forward")
        return original(name, globals, locals, fromlist, level)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", fail)
            importlib.reload(module)
            assert module.MARS is None
    finally:
        importlib.reload(module)
