import numpy as np

from UQPyL.surrogate.regression.polynomial_regression import PolynomialRegression


def test_polynomial_regression_fit_predict_smoke():
    # y = x^2
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    Y = (X**2)
    pr = PolynomialRegression(degree=2, onlyInteraction=False, lossType="Origin", fitIntercept=True)
    pr.fit(X, Y)
    pred = pr.predict(np.array([[4.0]]))
    assert pred.shape == (1, 1)
    assert np.isfinite(pred).all()

