import numpy as np
import pytest

from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.surrogate.base import MultiSurrogate


def test_linear_regression_origin_fit_predict():
    lr = LinearRegression(lossType="Origin", fitIntercept=True)
    x_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([[1.0], [3.0], [5.0]])  # y = 2x + 1
    lr.fit(x_train, y_train)

    y_pred = lr.predict(np.array([[3.0]]))
    assert y_pred.shape == (1, 1)
    assert np.allclose(y_pred, [[7.0]], atol=1e-6)


def test_multi_surrogate_fit_predict_and_type_check():
    m = MultiSurrogate(
        n_surrogates=2,
        models_list=[LinearRegression(lossType="Origin"), LinearRegression(lossType="Origin")],
    )

    x_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y1 = 2 * x_train + 1
    y2 = -1 * x_train + 0.5
    y_train = np.hstack([y1, y2])

    m.fit(x_train, y_train)
    y_pred = m.predict(np.array([[4.0], [5.0]]))
    assert y_pred.shape == (2, 2)

    with pytest.raises(ValueError):
        m.append(object())


