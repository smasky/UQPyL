"""校准无量纲指标的单位换算及真实退化情况。"""

import numpy as np
import pytest

from UQPyL.calibration import GLUE, SUFI2
from UQPyL.calibration import util
from UQPyL.problem import ModelProblem


metricReferences = {
    "nse": [0.98, 0.968],
    "r2": [0.98, 0.968],
    "pbias": [0.0, 8.0],
    "pearson_r": [0.9908470001860921, 1.0],
    "kge": [0.947873410771797, 0.92],
    "rfactor": 0.35777087639996635,
}


def metricInputs(name, scale, masked=False):
    obs = np.array([1.0, 2.0, 3.0, 4.0]) * scale
    sim = np.array([[1.1, 1.9, 3.2, 3.8], [1.2, 2.2, 3.2, 4.2]]) * scale
    if name == "rfactor":
        args = [obs, obs - 0.2 * scale, obs + 0.2 * scale]
    else:
        args = [obs, sim]
    if masked:
        args = [np.concatenate((value, np.full((*value.shape[:-1], 1), np.nan)), axis=-1) for value in args]
    return args


@pytest.mark.parametrize("name", metricReferences)
@pytest.mark.parametrize("scale", [1e-200, 1e-12, 1e-6, 1e6, 1e200])
def testDimensionlessMetricsMatchIndependentReferenceAfterUnitConversion(name, scale):
    args = metricInputs(name, scale)
    original = [value.copy() for value in args]
    actual = getattr(util, name)(*args)
    np.testing.assert_allclose(actual, metricReferences[name], rtol=3e-13, atol=3e-13)
    for value, before in zip(args, original):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("name", metricReferences)
def testSmallUnitsWithMaskedMissingValuesAndMultipleSimulationRows(name):
    args = metricInputs(name, 1e-12, masked=True)
    before = [value.copy() for value in args]
    mask = np.array([False, False, False, False, True])
    actual = getattr(util, name)(*args, mask=mask)
    np.testing.assert_allclose(actual, metricReferences[name], rtol=3e-13, atol=3e-13)
    for value, original in zip(args, before):
        np.testing.assert_array_equal(value, original)
    np.testing.assert_array_equal(mask, [False, False, False, False, True])


@pytest.mark.parametrize("name", ["nse", "r2", "pearson_r", "kge", "rfactor"])
def testTrueConstantObservationsRemainUndefinedInSmallUnits(name):
    obs = np.full(4, 2e-12)
    sim = np.arange(1.0, 5.0) * 1e-12
    args = (obs, sim - 0.1e-12, sim + 0.1e-12) if name == "rfactor" else (obs, sim)
    with pytest.raises(ValueError, match="variance|standard deviation"):
        getattr(util, name)(*args)


@pytest.mark.parametrize("name", ["pbias", "kge"])
def testGenuineZeroObservationSumOrMeanRemainsUndefined(name):
    obs = np.array([-7.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="sum|mean"):
        getattr(util, name)(obs, obs * 2)


@pytest.mark.parametrize("name,expected", [("pbias", 100.0), ("kge", 0.0)])
def testSmallNonzeroObservationMeanIsNotRejected(name, expected):
    np.testing.assert_allclose(getattr(util, name)([-1.0, 1.0, 1e-12], [-1.0, 1.0, 2e-12]), [expected], atol=2e-12)


@pytest.mark.parametrize("name,expected", [("pbias", 0.0), ("kge", 1.0)])
def testCancellationDoesNotHideANonzeroObservationSum(name, expected):
    # 精确和为 1；朴素浮点加和会丢失中间项。
    obs = np.array([1e20, 1.0, -1e20])
    np.testing.assert_allclose(getattr(util, name)(obs, obs), [expected], atol=3e-15)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def testCorrelationAllowsIndependentSimulationRowScales(scale):
    obs, sim = metricInputs("pearson_r", 1.0)
    sim[1] *= scale
    np.testing.assert_allclose(util.pearson_r(obs, sim), metricReferences["pearson_r"], atol=3e-15)


def testLargeFiniteKgeUsesFiniteRatiosWithoutSquaringOverflow():
    obs, sim = metricInputs("kge", 1.0)
    sim[1] *= 1e200
    expected = 1 - np.hypot(1e200 - 1, 1.08e200 - 1)
    np.testing.assert_allclose(util.kge(obs, sim), [metricReferences["kge"][0], expected], rtol=3e-15)


def testNseRetainsRepresentableVariationAroundALargeOffset():
    obs = 1e12 + np.arange(4.0)
    sim = obs + np.array([0.1, -0.1, 0.2, -0.2])
    expected = 1 - np.sum((sim - obs) ** 2) / 5
    np.testing.assert_allclose(util.nse(obs, sim), [expected], rtol=0, atol=2e-15)


@pytest.mark.parametrize("name", ["pearson_r", "kge"])
@pytest.mark.parametrize("constant", [0.1, 3e-12])
def testTrueConstantSimulationRowStillRaises(name, constant):
    obs = np.arange(1.0, 8.0) * 1e-12
    sim = np.vstack((obs * 1.1, np.full(7, constant)))
    with pytest.raises(ValueError, match="simulation variance"):
        getattr(util, name)(obs, sim)


@pytest.mark.parametrize("name", ["nse", "pearson_r", "kge", "rfactor"])
@pytest.mark.parametrize("constant", [0.1, 3e-12])
def testConstantDetectionDoesNotDependOnMeanRounding(name, constant):
    # 多个相同浮点数的 mean 可能与单个输入相差一个 ulp。
    obs = np.full(7, constant)
    sim = np.arange(1.0, 8.0) * constant
    args = (obs, sim - constant * 0.1, sim + constant * 0.1) if name == "rfactor" else (obs, sim)
    with pytest.raises(ValueError, match="variance|standard deviation"):
        getattr(util, name)(*args)


@pytest.mark.parametrize("methodClass", [GLUE, SUFI2])
def testNseCalibrationWorkflowIsInvariantToObservationUnits(methodClass):
    decisions = np.array([[-0.3], [0.0], [0.1], [0.4]])
    results = []
    for scale in [1.0, 1e-12]:
        observations = np.arange(1.0, 5.0).reshape(-1, 1) * scale

        def simulate(x):
            return (observations[None] + x[:, None, :] * scale).reshape(len(x), -1)

        problem = ModelProblem(nInput=1, lb=-1, ub=1, obs=(observations).reshape(-1), simFunc=simulate)
        kwargs = {"threshold": 0.95} if methodClass is GLUE else {"eliteSize": 2}
        result = methodClass(metric="nse", saveFlag=False).run(problem, decisions, **kwargs)
        np.testing.assert_allclose(result.diagnostics["scores"], [0.928, 1.0, 0.992, 0.872], atol=3e-15)
        np.testing.assert_allclose(result.bestDecs, [[0.0]])
        results.append(result)
    field = "behavioralDecs" if methodClass is GLUE else "eliteDecs"
    np.testing.assert_array_equal(getattr(results[0], field), getattr(results[1], field))
    if methodClass is SUFI2:
        assert results[0].diagnostics["rfactor"] == pytest.approx(results[1].diagnostics["rfactor"])
