import numpy as np
import pytest

from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Guass
from UQPyL.surrogate.kriging.kriging import regrpoly2
from UQPyL.surrogate.scaler import StandardScaler


def quadratic(X):
    x, y = X.T
    return (2 + 3 * x - 4 * y + 0.5 * x * x + 2 * x * y - y * y)[:, None]


def testQuadraticBasisMatchesExplicitTerms():
    X = np.array([[2.0, -3.0], [-1.0, 4.0], [0.0, 0.0]])
    x, y = X.T
    expected = np.column_stack((np.ones(len(X)), x, y, x * x, x * y, y * y))
    np.testing.assert_array_equal(regrpoly2(X), expected)


@pytest.mark.parametrize("scaled", [False, True])
def testQuadraticTrendReproducesKnownPolynomial(scaled):
    X = np.array([(x, y) for x in [-1.0, 0.0, 1.0] for y in [-1.0, 0.0, 1.0]])
    query = np.array([[-0.7, 0.3], [0.2, -0.4], [1.2, 0.8]])
    scalers = (StandardScaler(), StandardScaler()) if scaled else (None, None)
    model = KRG(regression="poly2", kernel=Guass(theta=0.7, theta_attr=None), scalers=scalers).fit(X, quadratic(X))
    mean, variance = model.predict(query, returnVar=True)
    meanWithStd, std = model.predict(query, returnStd=True)
    np.testing.assert_allclose(mean, quadratic(query), atol=1e-10)
    np.testing.assert_allclose(meanWithStd, mean, atol=1e-12)
    assert np.all(np.isfinite(variance)) and np.all(variance >= 0)
    np.testing.assert_allclose(variance, 0, atol=1e-20)
    np.testing.assert_allclose(std**2, variance, atol=1e-20)


def testQuadraticTrendUncertaintyMatchesIndependentDenseSolve():
    X = np.array([(x, y) for x in [-1.0, 0.0, 1.0] for y in [-1.0, 0.0, 1.0]])
    query = np.array([[-0.7, 0.3], [0.2, -0.4], [1.2, 0.8]])
    Y = quadratic(X) + np.sin(2 * X[:, :1] + X[:, 1:])
    theta = 0.7

    def basis(points):
        x, y = points.T
        return np.column_stack((np.ones(len(points)), x, y, x * x, x * y, y * y))

    def correlation(left, right):
        return np.exp(-theta * np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2))

    # Generalized least squares and universal-kriging variance, independently
    # of the production QR/Cholesky factors and trend implementation.
    design, queryDesign = basis(X), basis(query)
    covariance = correlation(X, X) + (10 + len(X)) * np.finfo(float).eps * np.eye(len(X))
    cross = correlation(X, query)
    solvedDesign = np.linalg.solve(covariance, design)
    normal = design.T @ solvedDesign
    beta = np.linalg.solve(normal, solvedDesign.T @ Y)
    residual = Y - design @ beta
    solvedResidual = np.linalg.solve(covariance, residual)
    sigma = float((residual.T @ solvedResidual).item()) / len(X)
    expectedMean = queryDesign @ beta + cross.T @ solvedResidual
    delta = queryDesign.T - design.T @ np.linalg.solve(covariance, cross)
    expectedVar = sigma * (
        1
        - np.sum(cross * np.linalg.solve(covariance, cross), axis=0)
        + np.sum(delta * np.linalg.solve(normal, delta), axis=0)
    )
    model = KRG(regression="poly2", kernel=Guass(theta=theta, theta_attr=None)).fit(X, Y)
    mean, variance = model.predict(query, returnVar=True)
    assert np.all(expectedVar > 0)
    np.testing.assert_allclose(mean, expectedMean, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(variance[:, 0], expectedVar, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case", ["too_few", "collinear"])
@pytest.mark.parametrize("optimize", [False, True])
def testQuadraticTrendRejectsUnidentifiableDesign(case, optimize):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    if case == "collinear":
        x = np.linspace(-1, 1, 8)
        X = np.column_stack((x, x))
    model = KRG(regression="poly2", kernel=Guass() if optimize else Guass(theta_attr=None))
    with pytest.raises(ValueError, match="(?i)(rank|regression|trend)"):
        model.fit(X, quadratic(X))
    assert not model.fitState
