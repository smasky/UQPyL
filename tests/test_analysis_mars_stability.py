"""High predictive R2 alone does not establish reliable MARS importance."""

import importlib
import warnings

import numpy as np
import pytest

from UQPyL.analysis import MARS
from UQPyL.analysis.runtime import AnaReader
from UQPyL.doe import LHS
from UQPyL.problem import Problem


def makeProblem(nInput=3, objFunc=None):
    return Problem(nInput=nInput, nObj=1, lb=-1, ub=1, objFunc=objFunc or (lambda x: x[:, :1]))


def testMarsWarnsAboutGcvSearchDespiteHighValidationR2():
    problem = makeProblem(5)
    x = LHS("classic").sample(problem, 512, seed=41)
    y = (3 * x[:, 0] + 2 * x[:, 1] * x[:, 2] * x[:, 3])[:, None]
    with pytest.warns(RuntimeWarning, match="GCV.*search"):
        result = MARS(verboseFlag=False, maxDegree=3).analyze(problem, x, y)
    diagnostic = result.extra["mars_validation"]["outputs"][0]
    assert diagnostic["validation_r2"] > 0.95
    assert diagnostic["gcv_search_unstable"]
    assert diagnostic["gcv_improvement_variable"] == 4
    assert diagnostic["gcv_improvement_fraction"] > 0.02
    # Keep the primary estimator, while explicitly flagging this known failure.
    assert result["S1_norm"].values[0, 0] > 0.98
    reference = np.array([3, 4 / 27, 4 / 27, 4 / 27, 0])
    reference /= reference.sum()
    assert max(abs(result["S1_norm"].values[0] - reference)) > 0.1


def testMarsRepeatedHoldoutsPreservePrimaryScoresAndModelCalls(tmp_path):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return 3 * x[:, :1] + x[:, 1:2]

    problem = makeProblem(objFunc=objective)
    problem.workDir = str(tmp_path)
    x = LHS("classic").sample(problem, 512, seed=17)
    primary = MARS(verboseFlag=False).analyze(problem, x)
    calls.clear()
    repeated = MARS(verboseFlag=False, saveFlag=True, nValidationRepeats=3).analyze(problem, x)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], x)
    for metric in primary.metrics:
        np.testing.assert_array_equal(repeated[metric.name].values, metric.values)
    stability = repeated.extra["mars_stability"]
    assert stability["split_seeds"] == [0, 1, 2]
    output = stability["outputs"][0]
    assert output["stable"] is True
    np.testing.assert_allclose(output["normalized_mean"], [0.9, 0.1, 0], atol=0.025)
    assert output["max_normalized_range"] < 0.05
    assert len(stability["splits"]) == 3
    assert primary.extra["mars_stability"]["outputs"][0]["stable"] is None
    with AnaReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
    assert loaded.extra["mars_stability"] == stability


def testMarsRepeatedWeightVariationMatchesIndependentStatistics(monkeypatch):
    module = importlib.import_module("UQPyL.analysis.methods.mars")

    class ControlledModel:
        fitCount = 0

        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            # Two fits with exact predictions but reversed GCV deletion losses.
            gcvSchedules = [[1, 9, 3, 1], [1, 3, 9, 1]]
            group, position = divmod(ControlledModel.fitCount, 4)
            self.gcv_ = gcvSchedules[group][position]
            ControlledModel.fitCount += 1
            self.coef = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]

        def predict(self, x):
            return np.column_stack([np.ones(len(x)), x]) @ self.coef

    monkeypatch.setattr(module, "MARSModel", ControlledModel)
    problem = makeProblem()
    x = LHS("classic").sample(problem, 100, seed=4)
    with pytest.warns(RuntimeWarning, match="normalized.*range"):
        result = MARS(verboseFlag=False, nValidationRepeats=2).analyze(problem, x, x[:, :1])
    np.testing.assert_allclose(result["S1_norm"].values, [[0.8, 0.2, 0]])
    output = result.extra["mars_stability"]["outputs"][0]
    assert output["stable"] is False
    np.testing.assert_allclose(output["normalized_mean"], [0.5, 0.5, 0])
    np.testing.assert_allclose(output["normalized_std"], [0.3, 0.3, 0])
    np.testing.assert_allclose(output["normalized_min"], [0.2, 0.2, 0])
    np.testing.assert_allclose(output["normalized_max"], [0.8, 0.8, 0])
    assert output["max_normalized_range"] == pytest.approx(0.6)
    assert ControlledModel.fitCount == 8


def testMarsRepeatedConstantOutputIsZeroWithoutWarnings():
    problem = makeProblem()
    x = LHS("classic").sample(problem, 100, seed=19)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = MARS(verboseFlag=False, nValidationRepeats=3).analyze(problem, x, np.full((100, 1), 1e300))
    assert emitted == []
    np.testing.assert_array_equal(result["S1_norm"].values, 0)
    output = result.extra["mars_stability"]["outputs"][0]
    assert output["constant_output"] and output["stable"]
    assert output["max_normalized_range"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nValidationRepeats": 0},
        {"nValidationRepeats": True},
        {"nValidationRepeats": 1.5},
        {"stabilityTolerance": 0},
        {"stabilityTolerance": np.inf},
        {"stabilityTolerance": 1.1},
        {"gcvImprovementTolerance": 0},
        {"gcvImprovementTolerance": np.nan},
    ],
)
def testMarsRejectsInvalidDiagnosticSettings(kwargs):
    with pytest.raises(ValueError):
        MARS(**kwargs)
