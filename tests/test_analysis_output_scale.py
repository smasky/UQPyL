import numpy as np
import pytest

from UQPyL.analysis import Sobol, FAST, RBDFAST
from UQPyL.doe import SaltelliDesign, FASTDesign, LHS
from UQPyL.problem import Problem


@pytest.fixture(params=[Sobol, FAST, RBDFAST])
def analysisCase(request):
    methodClass = request.param
    problem = Problem(nInput=2, nObj=1, lb=0., ub=1.,
                      objFunc=lambda X: X[:, :1] + 2 * X[:, 1:2])
    if methodClass is Sobol:
        design, count = SaltelliDesign(secondOrder=True), 1024
    elif methodClass is FAST:
        design, count = FASTDesign(M=4), 1025
    else:
        design, count = LHS("classic"), 1024
    X, meta = design.sampleWithMeta(problem, count, seed=1)
    method = methodClass(verboseFlag=False)
    options = {} if methodClass is RBDFAST else {"meta": meta}
    return method, problem, X, options


def analyze(case, Y):
    method, problem, X, options = case
    return method.analyze(problem, X, Y=Y, **options)


def assertSameMetrics(actual, expected, atol=2e-12):
    assert actual.metricNames == expected.metricNames
    for metric in expected.metrics:
        np.testing.assert_allclose(actual[metric.name].values, metric.values, rtol=2e-12, atol=atol)


@pytest.mark.parametrize("scale", [1e-200, -1e-200, 1e-10, -3., 1e200, -1e200])
def testNonzeroOutputScalingPreservesAllIndices(analysisCase, scale):
    X = analysisCase[2]
    Y = X[:, :1] + 2 * X[:, 1:2] + X[:, :1] * X[:, 1:2]
    expected = analyze(analysisCase, Y)
    scaled = Y * scale
    original = scaled.copy()
    scaled.flags.writeable = False
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = analyze(analysisCase, scaled)
    assertSameMetrics(result, expected)
    np.testing.assert_array_equal(scaled, original)
    np.testing.assert_array_equal(result.Y, original)


@pytest.mark.parametrize("value", [0., 1e-250, -3., 1e300])
def testExactlyConstantOutputRetainsZeroConvention(analysisCase, value):
    Y = np.full((len(analysisCase[2]), 1), value)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = analyze(analysisCase, Y)
    for metric in result.metrics:
        np.testing.assert_array_equal(metric.values, 0.)


@pytest.mark.parametrize("representation", ["subnormal", "largeOffset", "nearFloatLimit"])
def testRepresentableVariationSurvivesExtremeUnits(analysisCase, representation):
    X = analysisCase[2]
    Y = np.floor(32 * (X[:, :1] + 2 * X[:, 1:2])) - 48
    if representation == "subnormal":
        transformed = np.ldexp(Y, -1070)
    elif representation == "largeOffset":
        transformed = np.ldexp(Y, -10) + 2.**40
    else:
        transformed = np.ldexp(Y, 1017)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = analyze(analysisCase, transformed)
    assertSameMetrics(result, analyze(analysisCase, Y))


def testMultiOutputUsesIndependentScalesAndPreservesInput(analysisCase):
    X = analysisCase[2]
    Y = X[:, :1] + 2 * X[:, 1:2]
    expected = analyze(analysisCase, Y)
    outputs = np.column_stack((Y, Y * 1e-200, Y * -1e200, np.ones_like(Y)))
    original = outputs.copy()
    result = analyze(analysisCase, outputs)
    for metric in expected.metrics:
        values = result[metric.name].values
        for index in range(3):
            np.testing.assert_allclose(values[index], metric.values[0], rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(values[3], 0.)
    np.testing.assert_array_equal(outputs, original)
    np.testing.assert_array_equal(result.Y, original)


def testAdditiveModelRetainsKnownSensitivity(analysisCase):
    X = analysisCase[2]
    result = analyze(analysisCase, (X[:, :1] + 2 * X[:, 1:2]) * 1e-200)
    tolerance = .04 if isinstance(analysisCase[0], RBDFAST) else .003
    np.testing.assert_allclose(result["S1"].values, [[.2, .8]], atol=tolerance, rtol=0)
    if "ST" in result.metricNames:
        np.testing.assert_allclose(result["ST"].values, [[.2, .8]], atol=.003, rtol=0)
    if "S2" in result.metricNames:
        np.testing.assert_allclose(result["S2"].values, 0., atol=.003, rtol=0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def testNonfiniteOutputsAreRejected(analysisCase, value):
    Y = analysisCase[2][:, :1].copy()
    Y[3, 0] = value
    with pytest.raises(ValueError, match="finite"):
        analyze(analysisCase, Y)
