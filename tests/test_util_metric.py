import numpy as np
import pytest

from UQPyL.util.metric import r_square, nse, mse, rank_score, sort_score


def test_metric_r2_nse_mse_basic():
    y = np.array([[0.0], [1.0], [2.0]])
    yhat = y.copy()
    assert np.isclose(float(r_square(y, yhat)), 1.0)
    assert np.isclose(float(nse(y, yhat)), 1.0)
    assert np.allclose(mse(y, yhat), np.array([0.0]))


def test_metric_r2_warns_on_constant_target():
    y = np.array([[1.0], [1.0], [1.0]])
    yhat = np.array([[1.0], [2.0], [3.0]])
    with pytest.warns(RuntimeWarning):
        v = r_square(y, yhat)
    assert not np.isfinite(v)


def test_metric_rank_and_sort_scores_branches():
    # Perfect agreement
    y = np.array([[0.0], [1.0], [2.0], [3.0]])
    yhat = y.copy()
    assert np.isclose(float(rank_score(y, yhat)), 1.0)
    assert sort_score(y, yhat) == 0

    # Full reversal => negative rank score and non-zero sort_score
    yhat2 = y[::-1].copy()
    assert float(rank_score(y, yhat2)) < 0
    assert sort_score(y, yhat2) > 0

    # Includes ties (hits (ty==... and py==...) branch)
    yt = np.array([[0.0], [0.0], [1.0]])
    yp = np.array([[2.0], [2.0], [3.0]])
    v = rank_score(yt, yp)
    assert np.isfinite(v)

