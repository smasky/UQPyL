import numpy as np
import pytest

from UQPyL.calibration import util

METRICS = [util.mse, util.mae, util.rmse, util.nse, util.r2, util.pbias, util.pearson_r, util.kge]


@pytest.mark.parametrize("metric", METRICS, ids=lambda metric: metric.__name__)
@pytest.mark.parametrize("empty", [False, True], ids=["all-masked", "empty-input"])
def testMetricsRejectNoObservations(metric, empty):
    obs, sim, mask = ([], [], None) if empty else ([1, 2], [[1, 2]], [True, True])
    with pytest.raises(ValueError, match="No valid observations"):
        metric(obs, sim, mask=mask)


@pytest.mark.parametrize(
    "sim,mask,message",
    [
        (1.0, None, "1D or 2D"),
        (np.ones((1, 1, 2)), None, "1D or 2D"),
        ([1], None, "obs.size"),
        ([1, 2], [False], "mask size"),
    ],
)
def testSharedMetricPreparationRejectsInvalidShapes(sim, mask, message):
    with pytest.raises(ValueError, match=message):
        util.mse([1, 2], sim, mask=mask)


@pytest.mark.parametrize(
    "metric,obs,sim,message",
    [
        (util.nse, [2, 2], [2, 3], "variance"),
        (util.r2, [2, 2], [2, 3], "variance"),
        (util.pbias, [-1, 1], [-2, 2], "sum"),
        (util.pearson_r, [2, 2], [1, 2], "observation variance"),
        (util.pearson_r, [1, 2], [[1, 2], [3, 3]], "simulation variance"),
        (util.kge, [-1, 1], [-2, 2], "mean"),
        (util.kge, [2, 2], [1, 2], "variance"),
        (util.kge, [1, 2], [3, 3], "simulation variance"),
    ],
)
def testUndefinedMetricsRaiseExplicitErrors(metric, obs, sim, message):
    with pytest.raises(ValueError, match=message):
        metric(obs, sim)


@pytest.mark.parametrize("metric", METRICS, ids=lambda metric: metric.__name__)
def testMaskedMissingValuesMatchSubsetWithoutMutation(metric):
    obs = np.array([1.0, np.nan, 3.0])
    sim = np.array([[1.0, np.nan, 4.0], [2.0, np.inf, 3.0]])
    mask = np.array([False, True, False])
    originalObs, originalSim = obs.copy(), sim.copy()
    actual = metric(obs, sim, mask=mask)
    np.testing.assert_allclose(actual, metric([1, 3], [[1, 4], [2, 3]]))
    assert actual.shape == (2,)
    np.testing.assert_array_equal(obs, originalObs)
    np.testing.assert_array_equal(sim, originalSim)
    np.testing.assert_array_equal(mask, [False, True, False])


@pytest.mark.parametrize("metric", [util.pfactor, util.rfactor], ids=lambda metric: metric.__name__)
@pytest.mark.parametrize(
    "case,message",
    [
        ("empty", "No valid observations"),
        ("all_masked", "No valid observations"),
        ("length", "same size"),
        ("mask", "mask size"),
        ("reversed", "lower.*upper"),
    ],
)
def testIntervalMetricsRejectInvalidInputs(metric, case, message):
    obs, lower, upper, mask = [1, 3], [0, 2], [2, 4], None
    if case == "empty":
        obs, lower, upper = [], [], []
    elif case == "all_masked":
        mask = [True, True]
    elif case == "length":
        upper = [2]
    elif case == "mask":
        mask = [True]
    else:
        lower = [3, 2]
    with pytest.raises(ValueError, match=message):
        metric(obs, lower, upper, mask=mask)


def testIntervalMetricsMaskMissingValuesAndIncludeEndpoints():
    obs, lower, upper = [1.0, np.nan, 3.0, 5.0], [1.0, np.nan, 2.0, 6.0], [1.0, np.nan, 3.0, 7.0]
    mask = [False, True, False, False]
    assert util.pfactor(obs, lower, upper, mask=mask) == pytest.approx(2 / 3)
    assert util.rfactor(obs, lower, upper, mask=mask) == pytest.approx((2 / 3) / np.sqrt(8 / 3))


def testConstantObservationsAreValidForCoverageButNotNormalizedWidth():
    assert util.pfactor([2, 2], [1, 2], [2, 3]) == 1
    with pytest.raises(ValueError, match="standard deviation"):
        util.rfactor([2, 2], [1, 2], [2, 3])


def testZeroMeanStillAllowsLossEfficiencyAndCorrelation():
    for metric in (util.mse, util.mae, util.rmse, util.pearson_r):
        np.testing.assert_allclose(metric([-1, 1], [-2, 2]), [1])
    np.testing.assert_allclose(util.nse([-1, 1], [-2, 2]), [0])


def testMaskedReversedIntervalsAreIgnoredWithoutMutatingBounds():
    lower, upper = np.array([0.0, 9.0, 2.0]), np.array([2.0, -9.0, 4.0])
    originalLower, originalUpper = lower.copy(), upper.copy()
    assert util.pfactor([1, 2, 3], lower, upper, mask=[False, True, False]) == 1
    assert util.rfactor([1, 2, 3], lower, upper, mask=[False, True, False]) == 2
    np.testing.assert_array_equal(lower, originalLower)
    np.testing.assert_array_equal(upper, originalUpper)
