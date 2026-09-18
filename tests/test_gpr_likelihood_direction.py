import numpy as np
import pytest
from scipy.stats import multivariate_normal

from UQPyL.optimization.soea import GA
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF


def negativeLogDensity(X, Y, lengthScale, noise):
    distances = (X[:, None, :] - X[None, :, :]) / lengthScale
    covariance = np.exp(-0.5 * np.sum(distances**2, axis=2))
    covariance += noise * np.eye(len(X))
    return -np.sum(multivariate_normal.logpdf(Y.T, cov=covariance))


@pytest.mark.parametrize("nOutputs", [1, 2])
def test_fixed_gpr_records_negative_joint_log_density(nOutputs):
    X = np.linspace(0, 1, 12).reshape(-1, 1)
    Y = np.column_stack([np.sin(6 * X[:, 0]), np.cos(3 * X[:, 0])])[:, :nOutputs]
    model = GPR(kernel=RBF(length_scale=0.3, length_attr=None), C=1e-6, C_attr=None)
    model.fit(X, Y)

    expected = negativeLogDensity(X, Y, 0.3, 1e-6)
    assert model.fitState["objective"] == pytest.approx(expected, rel=1e-7)
    assert model._objfunc(X, Y) == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("optimizerName", ["Boxmin", "LBFGSB", "GA"])
@pytest.mark.parametrize("logScale", [False, True])
def test_internal_optimization_improves_log_density(optimizerName, logScale):
    X = np.linspace(0, 1, 12).reshape(-1, 1)
    Y = np.sin(6 * X)
    optimizer = optimizerName
    if optimizerName == "GA":
        optimizer = GA(nPop=16, maxFEs=192, tolerate=None,
                       verboseFlag=False, logFlag=False, saveFlag=False)
    lengthAttr = {"lb": 0.1, "ub": 3.0, "type": "float", "log": logScale}
    model = GPR(kernel=RBF(length_scale=0.3, length_attr=lengthAttr),
                C=1e-6, C_attr=None, optimizer=optimizer, nRestartTimes=1)
    model.rng = np.random.default_rng(2)
    model.fit(X, Y)

    lengthScale = float(np.asarray(model.setting.get("l")).item())
    objective = negativeLogDensity(X, Y, lengthScale, 1e-6)
    baseline = negativeLogDensity(X, Y, 0.3, 1e-6)
    assert objective < baseline
    assert model.fitState["objective"] == pytest.approx(objective, rel=1e-7)
    assert np.sqrt(np.mean((model.predict(X) - Y)**2)) < 1e-3
