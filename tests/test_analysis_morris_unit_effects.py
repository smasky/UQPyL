"""Standard Morris effects use parameter-range-relative input steps."""

import numpy as np
import pytest

from UQPyL.analysis import Morris
from UQPyL.analysis.runtime import AnaReader
from UQPyL.doe import MorrisDesign
from UQPyL.problem import Problem


META = dict(designType="morris", numLevels=4)


def makeProblem():
    return Problem(nInput=3, nObj=1, lb=[10, -2, 100], ub=[30, 1, 105], objFunc=lambda x: 3 * x[:, :1] - 2 * x[:, 1:2])


def testMorrisUnitEffectsMatchAnalyticalLinearEffects():
    problem = makeProblem()
    x, meta = MorrisDesign().sampleWithMeta(problem, 12, seed=17)
    y = problem.evaluate(x).objs
    unit = Morris(verboseFlag=False).analyze(problem, x, y, meta)
    np.testing.assert_allclose(unit["mu"].values, [[60, -6, 0]], atol=1e-13)
    np.testing.assert_allclose(unit["S1_norm"].values, [[10 / 11, 1 / 11, 0]], atol=1e-14)
    assert unit.extra["morris_effects"] == dict(effect_mode="unit", effect_units="output", input_ranges=[20, 3, 5])


@pytest.mark.parametrize("factor", [0.001, 1000.0])
def testMorrisUnitEffectsIgnoreInputUnitsAndOrigin(factor):
    problem = makeProblem()
    x, meta = MorrisDesign().sampleWithMeta(problem, 12, seed=3)
    y = problem.evaluate(x).objs
    converted = x.copy()
    converted[:, 0] = x[:, 0] * factor + 7
    lb, ub = problem.lb.astype(float), problem.ub.astype(float)
    lb[0, 0], ub[0, 0] = lb[0, 0] * factor + 7, ub[0, 0] * factor + 7
    convertedProblem = Problem(nInput=3, nObj=1, lb=lb, ub=ub, objFunc=lambda x: x[:, :1])
    expected = Morris(verboseFlag=False).analyze(problem, x, y, meta)
    actual = Morris(verboseFlag=False).analyze(convertedProblem, converted, y, meta)
    for name in ("mu", "mu_star", "sigma", "S1_norm"):
        np.testing.assert_allclose(actual[name].values, expected[name].values, atol=1e-10)


@pytest.mark.parametrize("scale", [1.0, 1e-200, -1e200])
def testMorrisUnitNonlinearMeanAndSampleSigma(scale):
    # The signed unit steps yield x0 effects 2, 6 and x1 effects 12, 12.
    x = np.array([[0, 0], [1, 0], [1, 2], [2, 4], [1, 4], [1, 2]], dtype=float)
    problem = Problem(nInput=2, nObj=1, lb=[0, 0], ub=[2, 4], objFunc=lambda x: x[:, :1])
    y = (x[:, :1] ** 2 + 3 * x[:, 1:2]) * scale
    result = Morris(verboseFlag=False).analyze(problem, x, y, META)
    np.testing.assert_allclose(result["mu"].values / scale, [[4, 12]], atol=1e-13)
    np.testing.assert_allclose(result["mu_star"].values / abs(scale), [[4, 12]], atol=1e-13)
    np.testing.assert_allclose(result["sigma"].values / abs(scale), [[np.sqrt(8), 0]], atol=1e-13)
    np.testing.assert_allclose(result["S1_norm"].values, [[0.25, 0.75]], atol=1e-14)
    np.testing.assert_array_equal(result.X, x)
    np.testing.assert_array_equal(result.Y, y)


def testMorrisDecodesSamplesOnceAndPersistsEffectUnits(tmp_path):
    problem = makeProblem()
    problem.workDir = str(tmp_path)
    real, realMeta = MorrisDesign().sampleWithMeta(problem, 10, seed=7)
    unit, unitMeta = MorrisDesign().sampleWithMeta(problem, 10, seed=7, output="unit")
    expected = Morris(verboseFlag=False).analyze(problem, real, meta=realMeta)
    actual = Morris(verboseFlag=False, saveFlag=True).analyze(problem, unit, meta=unitMeta)
    np.testing.assert_allclose(actual["mu"].values, expected["mu"].values, atol=1e-13)
    np.testing.assert_allclose(actual.X, real, atol=1e-13)
    assert unitMeta["output"] == "unit"
    with AnaReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
    assert loaded.extra["morris_effects"] == actual.extra["morris_effects"]


def testMorrisUsesPhysicalDiscreteRange():
    problem = Problem(
        nInput=2, nObj=1, lb=[0, 0], ub=[1, 4], varType=[2, 0], varSet={0: [10, 15, 30]}, objFunc=lambda x: x[:, :1]
    )
    x = np.array([[10, 0], [15, 0], [15, 2], [30, 4], [15, 4], [15, 2]], dtype=float)
    y = 2 * x[:, :1] + 3 * x[:, 1:2]
    result = Morris(verboseFlag=False).analyze(problem, x, y, META)
    np.testing.assert_allclose(result["mu"].values, [[40, 12]], atol=1e-13)
    assert result.extra["morris_effects"]["input_ranges"] == [20, 4]


def testMorrisInteractionEffectsMatchHandDerivedTrajectories():
    # x0*x1 gives unit effects x0=[0, 8], x1=[4, 4].
    x = np.array([[0, 0], [1, 0], [1, 2], [2, 4], [1, 4], [1, 2]], dtype=float)
    problem = Problem(nInput=2, nObj=1, lb=[0, 0], ub=[2, 4], objFunc=lambda x: x[:, :1] * x[:, 1:2])
    result = Morris(verboseFlag=False).analyze(problem, x, meta=META)
    np.testing.assert_allclose(result["mu"].values, [[4, 4]], atol=1e-13)
    np.testing.assert_allclose(result["mu_star"].values, [[4, 4]], atol=1e-13)
    np.testing.assert_allclose(result["sigma"].values, [[np.sqrt(32), 0]], atol=1e-13)
    np.testing.assert_allclose(result["S1_norm"].values, [[0.5, 0.5]], atol=1e-14)


@pytest.mark.parametrize("ub", [[0, 4], [np.inf, 4]])
def testMorrisRequiresFinitePositiveRanges(ub):
    problem = Problem(nInput=2, nObj=1, lb=[0, 0], ub=ub, objFunc=lambda x: x[:, :1])
    x = np.array([[0, 0], [1, 0], [1, 2], [2, 4], [1, 4], [1, 2]], dtype=float)
    with pytest.raises(ValueError, match="finite, positive"):
        Morris(verboseFlag=False).analyze(problem, x, x[:, :1], META)


@pytest.mark.parametrize("dtype", [np.float64, np.uint8, np.bool_])
def testMorrisThresholdEffectsPreserveDirectionAcrossOutputTypes(dtype):
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: (x > 0.5).astype(np.uint8))
    x, meta = MorrisDesign().sampleWithMeta(problem, 8, seed=17)
    y = (x > 0.5).astype(dtype)
    originalX, originalY = x.copy(), y.copy()
    result = Morris(verboseFlag=False).analyze(problem, x, y, meta)
    # Each signed unit step is +/-2/3 and crosses the threshold.
    np.testing.assert_allclose(result["mu"].values, [[1.5]])
    np.testing.assert_allclose(result["mu_star"].values, [[1.5]])
    np.testing.assert_allclose(result["sigma"].values, [[0.0]])
    np.testing.assert_array_equal(x, originalX)
    np.testing.assert_array_equal(y, originalY)
    assert result.Y.dtype == y.dtype


def testMorrisEvaluatedUnsignedOutputMatchesAnalyticalEffects():
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: (x > 0.5).astype(np.uint8))
    x, meta = MorrisDesign().sampleWithMeta(problem, 8, seed=17)
    result = Morris(verboseFlag=False).analyze(problem, x, meta=meta)
    np.testing.assert_allclose(result["mu"].values, [[1.5]])
    np.testing.assert_allclose(result["mu_star"].values, [[1.5]])
    np.testing.assert_allclose(result["sigma"].values, [[0.0]])
    assert result.Y.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.float64, np.uint8])
def testMorrisIntegerInputUsesSignedRangeRelativeSteps(dtype):
    problem = Problem(nInput=1, nObj=1, lb=0, ub=2, varType=[1], objFunc=lambda x: x.astype(float))
    x, meta = MorrisDesign().sampleWithMeta(problem, 8, seed=17)
    x = x.astype(dtype)
    result = Morris(verboseFlag=False).analyze(problem, x, x.astype(float), meta)
    np.testing.assert_allclose(result["mu"].values, [[2.0]])
    np.testing.assert_allclose(result["mu_star"].values, [[2.0]])
    np.testing.assert_allclose(result["sigma"].values, [[0.0]])
    assert result.X.dtype == x.dtype


@pytest.mark.parametrize("numTrajectory", [0, 1])
def testMorrisRequiresEnoughTrajectoriesForSampleSigma(numTrajectory):
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    x, meta = MorrisDesign().sampleWithMeta(problem, 1, seed=3)
    if numTrajectory == 0:
        x = x[:0]
    with pytest.raises(ValueError, match="at least two trajectories"):
        Morris(verboseFlag=False).analyze(problem, x, x, meta)
