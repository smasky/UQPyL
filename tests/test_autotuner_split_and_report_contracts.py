"""Autotuner split and report contracts.

Migrated from test_review_c16_c18_c19_c21.py; original regression provenance is retained below.
"""

import json
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as GpRbf
from UQPyL.surrogate.scaler import StandardScaler
from UQPyL.surrogate.split import KFold

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


# Regression source: test_review_c16_c18_c19_c21.py::testTunerRejectsMultiobjectiveOptimizerBeforeFitting
def testTunerRejectsMultiobjectiveOptimizerBeforeFitting(monkeypatch):
    model = LinearRegression(lossType="Ridge")

    def fail(*args):
        pytest.fail("Unsupported optimizer must be rejected before preprocessing.")

    monkeypatch.setattr(model, "prepareTrainingData", fail)
    with pytest.raises(ValueError, match="objective"):
        AutoTuner(model, NSGAII(**QUIET)).optTune(np.arange(10), np.arange(10), paraList=["C"])


class EnumeratingOptimizer:
    def run(self, problem, seed):
        candidates = np.array([[-4.0], [-2.0]])
        scores = problem.evaluate(candidates).objs
        best = np.argmax(scores[:, 0])
        return SimpleNamespace(bestDecs=candidates[best : best + 1], bestObjs=scores[best : best + 1])


def tune(tuner, entry, x, y, **kwargs):
    parameters = {"paraGrid": {"C": [-4.0, -2.0]}} if entry == "gridTune" else {"paraList": ["C"]}
    return getattr(tuner, entry)(x, y, tuneMode="joint", seed=3, **parameters, **kwargs)


# Regression source: test_review_c16_c18_c19_c21.py::testExplicitSplitIsUsedWithoutPreprocessingLeakage
@pytest.mark.parametrize("entry", ["gridTune", "optTune"])
@pytest.mark.parametrize("splitKind", ["fixed", "object", "callable", "rng"])
def testExplicitSplitIsUsedWithoutPreprocessingLeakage(entry, splitKind):
    class TrackingScaler(StandardScaler):
        def __init__(self):
            super().__init__()
            self.seen = []

        def fit(self, x):
            self.seen.append(x.copy())
            return super().fit(x)

    x = np.array([0.0, 1, 2, 3, 4, 5, 100, 200])[:, None]
    y = 2 * x + x * x
    train, validation = np.arange(5), np.array([6, 7])  # Index 5 is a deliberate time gap.

    class Splitter:
        def split(self, data, *, seed):
            assert isinstance(seed, int)
            data[:] = -99
            return train, validation

    options = {
        "fixed": {"splitIndices": (train, validation)},
        "object": {"splitter": Splitter()},
        "callable": {"splitter": lambda data: (train, validation)},
        "rng": {"splitter": lambda data, rng: (train, validation)},
    }[splitKind]
    originalX, originalY = x.copy(), y.copy()
    model = LinearRegression(lossType="Ridge", scalers=(TrackingScaler(), TrackingScaler()))
    tuner = AutoTuner(model, EnumeratingOptimizer())
    result = tune(tuner, entry, x, y, **options)
    reference = AutoTuner(
        LinearRegression(lossType="Ridge", scalers=(StandardScaler(), StandardScaler())), EnumeratingOptimizer()
    )
    expected = tune(reference, entry, x, y, splitIndices=(train, validation))
    for actual, target in zip(result, expected):
        np.testing.assert_allclose(actual, target)
    np.testing.assert_allclose(model.predict(x), reference.model.predict(x))
    for scaler, values in [(model.xScaler, x), (model.yScaler, y)]:
        assert len(scaler.seen) == 2
        np.testing.assert_array_equal(scaler.seen[0], values[train])
        np.testing.assert_array_equal(scaler.seen[1], values)
    np.testing.assert_array_equal(tuner.lastSplit["train_indices"], train)
    np.testing.assert_array_equal(tuner.lastSplit["test_indices"], validation)
    np.testing.assert_array_equal(x, originalX)
    np.testing.assert_array_equal(y, originalY)
    report = tuner.getReport()
    assert (report["training_samples"], report["validation_samples"], report["refit_samples"]) == (5, 2, 8)
    assert report["fit_calls"] == 3
    assert len(report["candidates"]) == 2
    json.dumps(report)
    train[0] = 4
    assert tuner.lastSplit["train_indices"][0] == 0


# Regression source: test_review_c16_c18_c19_c21.py::testBadValidationSplitFailsBeforeFitting
@pytest.mark.parametrize("entry", ["gridTune", "optTune"])
@pytest.mark.parametrize(
    "indices",
    [
        ([], [3, 4]),
        ([0, 1], []),
        ([0, 0], [3, 4]),
        ([0, 1], [1, 4]),
        ([-1, 0], [3, 4]),
        ([0, 1], [3, 10]),
        ([0.0, 1.0], [3, 4]),
        ([False, True], [3, 4]),
        ([[0, 1]], [3, 4]),
        ([0, 1], [3]),
        ([0, 1], [3, 3]),
    ],
)
def testBadValidationSplitFailsBeforeFitting(entry, indices, monkeypatch):
    model = LinearRegression(lossType="Ridge")

    def fail(*args):
        pytest.fail("Invalid split must fail before preprocessing.")

    monkeypatch.setattr(model, "prepareTrainingData", fail)
    tuner = AutoTuner(model, EnumeratingOptimizer())
    with pytest.raises(ValueError):
        tune(tuner, entry, np.arange(10), np.arange(10), splitIndices=indices)
    assert tuner.getReport()["status"] == "failed"
    assert tuner.getReport()["fit_calls"] == 0


# Regression source: test_review_c16_c18_c19_c21.py::testSplitAmbiguityAndMultipleFoldsAreExplicit
def testSplitAmbiguityAndMultipleFoldsAreExplicit():
    tuner = AutoTuner(LinearRegression(lossType="Ridge"))
    x = np.arange(20.0)[:, None]
    with pytest.raises(ValueError, match="only one"):
        tune(tuner, "gridTune", x, x * x, splitter=lambda data: None, splitIndices=([0, 1], [3, 4]))
    with pytest.raises(ValueError, match="single fold"):
        tune(tuner, "gridTune", x, x * x, splitter=KFold(4))
    train, validation = KFold(4).split(x, seed=1)
    tune(tuner, "gridTune", x, x * x, splitIndices=(train[0], validation[0]))


# Regression source: test_review_c16_c18_c19_c21.py::testTuningReportSeparatesCandidateFitAndFullRefitWithoutExtraCalls
@pytest.mark.parametrize("mode", ["joint", "separate"])
def testTuningReportSeparatesCandidateFitAndFullRefitWithoutExtraCalls(mode):
    x = np.linspace(0, 1, 24)[:, None]
    y = np.sin(6 * x)
    # Keep candidates outside an explicit domain to test internal bound enforcement.
    model = GPR(
        kernel=GpRbf(length_attr={"lb": 1, "ub": 1e5, "type": "float", "log": True}),
        C_attr=None,
        nRestartTimes=0,
    )
    original = model._objfunc
    observed = []

    def objective(*args, **kwargs):
        observed.append(1)
        return original(*args, **kwargs)

    model._objfunc = objective
    tuner = AutoTuner(model)
    best, score = tuner.gridTune(x, y, {"l": np.log([0.1, 0.8])}, ratio=25, seed=1, tuneMode=mode)
    report = tuner.getReport()
    assert model._objfunc is objective
    assert report["status"] == "finished" and report["fit_calls"] == 3
    assert report["tracked_objective_evaluations"] == len(observed)
    assert report["best_validation_score"] == score
    for candidate, start in zip(report["candidates"], [0.1, 0.8]):
        np.testing.assert_allclose(candidate["candidate_encoded"]["l"], np.log(start))
        np.testing.assert_allclose(candidate["fit"]["parameters_before_fit"]["l"], [start])
        assert candidate["validation_score"] is not None
        assert candidate["fit"]["elapsed_seconds"] >= 0
        fitted = candidate["fit"]["parameters_after_fit"]["l"]
        if mode == "joint":
            np.testing.assert_allclose(fitted, [start])
        else:
            assert (
                float(np.asarray(fitted).item()) >= 1
            )  # Internal optimizer enforces the kernel's declared lower bound.
    assert len(observed) == 3 if mode == "joint" else len(observed) > 3
    np.testing.assert_allclose(report["final_parameters"]["l"], best)
    saved = deepcopy(report)
    report["candidates"][0]["fit"]["parameters_after_fit"]["l"] = -99
    assert tuner.getReport() == saved
    assert model.predict(x).shape == y.shape


# Regression source: test_review_c16_c18_c19_c21.py::testReportRecordsFailuresAndResetsOnNextCall
@pytest.mark.parametrize("stage", ["numerical", "refit", "interrupt"])
def testReportRecordsFailuresAndResetsOnNextCall(stage, monkeypatch):
    x = np.arange(12.0)[:, None]
    model = LinearRegression(lossType="Ridge")
    tuner = AutoTuner(model)
    original = model.fitModel
    count = 0

    def fit(a, b):
        nonlocal count
        count += 1
        if stage == "numerical" and count == 1:
            raise np.linalg.LinAlgError("candidate failure")
        if stage == "refit" and len(a) == len(x):
            raise ValueError("refit failure")
        if stage == "interrupt":
            raise KeyboardInterrupt()
        return original(a, b)

    with monkeypatch.context() as patch:
        patch.setattr(model, "fitModel", fit)
        if stage == "numerical":
            tune(tuner, "gridTune", x, x * x, ratio=25)
            assert tuner.getReport()["candidates"][0]["status"] == "failed"
            assert tuner.getReport()["best_candidate_index"] == 1
        else:
            with pytest.raises(KeyboardInterrupt if stage == "interrupt" else ValueError):
                tune(tuner, "gridTune", x, x * x, ratio=25)
            assert tuner.getReport()["status"] == ("interrupted" if stage == "interrupt" else "failed")
            with pytest.raises(RuntimeError):
                model.predict(x)
            if stage == "refit":
                assert tuner.getReport()["final_refit"]["status"] == "failed"
    old = tuner.getReport()
    saved = deepcopy(old)
    tune(tuner, "gridTune", x, x * x, ratio=25)
    assert old == saved
    assert len(tuner.getReport()["candidates"]) == 2
    assert tuner.candidateFailures == []
