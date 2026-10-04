"""Independent interpolation, integer Lasso, and valid nu-SVR searches."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from UQPyL.surrogate import AutoTuner
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.regression import LinearRegression, PolynomialRegression
from UQPyL.surrogate.svr import SVR
from UQPyL.optimization.soea import GA


@pytest.mark.parametrize("scale", [1.0, 1000.0, 10000.0])
@pytest.mark.parametrize("linear", [True, False])
def test_cubic_rbf_preserves_solution_under_input_units(scale, linear):
    x = np.array([0.0, 0.1, 0.27, 0.46, 0.65, 0.82, 1.0])[:, None]
    y = 2 + 3 * x + (0 if linear else np.sin(5 * x))
    query = np.array([0.07, 0.21, 0.53, 0.91])[:, None]
    expected = CubicSpline(x.ravel(), y.ravel(), bc_type="natural")(query.ravel())
    model = RBF().fit(x * scale, y)
    np.testing.assert_allclose(model.predict(query * scale).ravel(), expected, rtol=1e-11, atol=1e-11)


def test_singular_rbf_warns_and_reports_conflicting_observations():
    model = RBF().fit(np.arange(4.0)[:, None], np.arange(4.0)[:, None])
    with pytest.warns(RuntimeWarning, match="singular"):
        model.fit(np.array([[0.0], [0.0], [1.0]]), np.array([[0.0], [2.0], [1.0]]))
    np.testing.assert_allclose(model.predict([[0.0], [0.5], [1.0]]), 1.0, atol=1e-12)
    diagnostic = model.fitState["linearSolve"]
    assert diagnostic["method"] == "constrained_lstsq"
    assert diagnostic["relativeTrainingResidual"] == pytest.approx(np.sqrt(2 / 5))
    assert diagnostic["repeatedInputs"]
    model.fit(np.arange(4.0)[:, None], np.arange(4.0)[:, None])
    assert model.fitState["linearSolve"] == {"method": "lu"}


@pytest.mark.parametrize("modelClass", [LinearRegression, PolynomialRegression])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("xType,yType", [(np.int64, np.int64), (np.float32, np.int64), (np.int64, np.float64)])
def test_lasso_integer_inputs_match_float_without_mutation(modelClass, intercept, xType, yType):
    x = np.arange(8).reshape(-1, 1).astype(xType)
    y = (2 * np.arange(8) + 1).reshape(-1, 1).astype(yType)
    originalX, originalY = x.copy(), y.copy()
    x.flags.writeable = y.flags.writeable = False
    model = modelClass(lossType="Lasso", fitIntercept=intercept).fit(x, y)
    expected = modelClass(lossType="Lasso", fitIntercept=intercept).fit(x.astype(float), y.astype(float))
    np.testing.assert_allclose(model.predict(x), expected.predict(x), rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(x, originalX)
    np.testing.assert_array_equal(y, originalY)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_nu_svr_default_search_stays_valid(seed):
    x = np.linspace(-1, 1, 30)[:, None]
    model = SVR(symbol="nu-SVR", C=1)
    optimizer = GA(nPop=8, maxFEs=8, verboseFlag=False, saveFlag=False, logFlag=False)
    tuner = AutoTuner(model, optimizer)
    nu, score = tuner.optTune(x, x * x + x, paraList=["nu"], seed=seed, ratio=30, tuneMode="joint")
    assert 0 < nu <= 1 and np.all(np.isfinite(score))
    assert tuner.lastReport["status"] == "finished"
    for candidate in tuner.lastReport["candidates"]:
        assert candidate["status"] == "finished"


@pytest.mark.parametrize("nu", [0.0, -1.0, 1.1, np.nan, np.inf])
def test_nu_svr_rejects_invalid_parameters_before_backend(nu, monkeypatch):
    import UQPyL.surrogate.svr.support_vector_machine as module

    def unexpected(*args):
        pytest.fail("invalid nu reached native training")

    monkeypatch.setattr(module, "svm_fit", unexpected)
    with pytest.raises(ValueError, match="nu"):
        SVR(symbol="nu-SVR", nu=nu).fit(np.arange(5.0)[:, None], np.arange(5.0)[:, None])
