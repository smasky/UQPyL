"""MARS must identify interactions or warn about unreliable screening."""

import numpy as np
import pytest
from UQPyL.analysis import MARS
from UQPyL.doe import LHS
from UQPyL.problem import Problem


def makeProblem(nInput=3):
    return Problem(nInput=nInput, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])


@pytest.mark.parametrize("seed", [3, 17, 73])
@pytest.mark.parametrize("mixed", [False, True])
def testMarsRecoversInteractionInsteadOfRankingInactiveInput(seed, mixed):
    p = makeProblem()
    x = LHS("classic").sample(p, 512, seed=seed)
    y = (x[:, 0] * x[:, 1] + (0.1 * x[:, 2] if mixed else 0))[:, None]
    result = MARS(verboseFlag=False).analyze(p, x, y)
    weights = result["S1_norm"].values[0]
    assert min(weights[:2]) > 0.4
    assert weights[2] < 0.1
    assert result.extra["mars_validation"]["outputs"][0]["validation_r2"] > 0.99


def testMarsWarnsAndReturnsUnlearnedInteractionWhenConfiguredAsAdditive():
    p = makeProblem()
    x = LHS("classic").sample(p, 512, seed=3)
    with pytest.warns(RuntimeWarning, match="minValidationR2=0.8"):
        result = MARS(verboseFlag=False, maxDegree=1).analyze(p, x, (x[:, 0] * x[:, 1])[:, None])
    assert result["S1"].values.shape == (1, 3)
    assert result.extra["mars_validation"]["outputs"][0]["validation_r2"] < 0.8


def testMarsWarnsAndReturnsNoiseImportance():
    p = makeProblem()
    x = LHS("classic").sample(p, 256, seed=17)
    y = np.random.default_rng(25).normal(size=(256, 1))
    with pytest.warns(RuntimeWarning, match="validation R2"):
        result = MARS(verboseFlag=False).analyze(p, x, y)
    assert np.all(np.isfinite(result["S1"].values))


def testMarsConstantAndSingleInputAreDefined():
    p = makeProblem(1)
    x = np.linspace(-1, 1, 100)[:, None]
    method = MARS(verboseFlag=False)
    constant = method.analyze(p, x, np.ones_like(x))
    np.testing.assert_array_equal(constant["S1"].values, [[0]])
    assert constant.extra["mars_validation"]["outputs"][0]["constant_output"]
    active = method.analyze(p, x, 3 * x)
    np.testing.assert_allclose(active["S1_norm"].values, [[1]])
    assert active.extra["mars_validation"]["outputs"][0]["validation_r2"] > 0.99
    assert constant.extra["mars_validation"]["outputs"][0]["constant_output"]


@pytest.mark.parametrize(
    "kwargs",
    [{"maxDegree": 0}, {"maxDegree": True}, {"maxTerms": 1.5}, {"minValidationR2": 0}, {"minValidationR2": np.nan}],
)
def testMarsRejectsInvalidSettings(kwargs):
    with pytest.raises(ValueError):
        MARS(**kwargs)


def testMarsRequiresEnoughRowsForValidation():
    x = np.zeros((10, 3))
    with pytest.raises(ValueError, match="at least 20"):
        MARS(verboseFlag=False).analyze(makeProblem(), x, np.zeros((10, 1)))


def testMarsDoesNotConvertImprovedReducedGcvIntoImportance(monkeypatch):
    import importlib

    module = importlib.import_module("UQPyL.analysis.methods.mars")
    fittedRows = []

    class ControlledModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            fittedRows.append(len(x))
            self.coef = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
            self.gcv_ = 2.0 if x.shape[1] == 3 else 1.0

        def predict(self, x):
            return np.column_stack((np.ones(len(x)), x)) @ self.coef

    monkeypatch.setattr(module, "MARSModel", ControlledModel)
    p = makeProblem()
    x = LHS("classic").sample(p, 100, seed=4)
    with pytest.warns(RuntimeWarning, match="GCV"):
        result = MARS(verboseFlag=False).analyze(p, x, x[:, :1] + 2 * x[:, 1:2])
    np.testing.assert_array_equal(result["S1"].values, [[0, 0, 0]])
    np.testing.assert_array_equal(result["S1_norm"].values, [[0, 0, 0]])
    assert fittedRows == [80] * 4


@pytest.mark.parametrize("r2, shouldWarn", [(0.79, True), (0.85, False)])
def testMarsDefaultWarningThresholdIsPointEight(monkeypatch, r2, shouldWarn):
    import importlib
    import warnings

    module = importlib.import_module("UQPyL.analysis.methods.mars")

    class ControlledModel:
        def __init__(self, **kwargs):
            self.gcv_ = 1.0

        def fit(self, x, y):
            self.coef = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]

        def predict(self, x):
            actual = np.column_stack((np.ones(len(x)), x)) @ self.coef
            return actual + np.sqrt((1 - r2) * np.var(actual))

    monkeypatch.setattr(module, "MARSModel", ControlledModel)
    x = LHS("classic").sample(makeProblem(), 100, seed=19)
    method = MARS(verboseFlag=False)
    assert method.get("minValidationR2") == 0.8
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = method.analyze(makeProblem(), x, x[:, :1])
    assert len(emitted) == int(shouldWarn)
    if shouldWarn:
        assert emitted[0].category is RuntimeWarning
    assert result.extra["mars_validation"]["outputs"][0]["validation_r2"] == pytest.approx(r2)
