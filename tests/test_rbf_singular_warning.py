"""Singular fits expose their approximation and retain trend constraints."""

import numpy as np
import pytest
from scipy.interpolate import RBFInterpolator
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.rbf.kernel import Cubic, ThinPlateSpline, Linear, Gaussian, Multiquadric


@pytest.mark.parametrize(
    "kernelClass,name,degree",
    [
        (Cubic, "cubic", 1),
        (ThinPlateSpline, "thin_plate_spline", 1),
        (Linear, "linear", 0),
        (Gaussian, "gaussian", -1),
        (Multiquadric, "multiquadric", 0),
    ],
)
@pytest.mark.parametrize("conflict", [False, True])
def test_duplicate_observations_match_independent_averaged_interpolation(kernelClass, name, degree, conflict):
    uniqueX = np.array([[0.0], [0.3], [0.7], [1.0]])
    uniqueY = np.sin(3 * uniqueX)
    x = np.vstack([uniqueX, uniqueX[[1]]])
    y = np.vstack([uniqueY, uniqueY[[1]] + (2 if conflict else 0)])
    averagedY = uniqueY.copy()
    averagedY[1] += 1 if conflict else 0
    query = np.linspace(0, 1, 11)[:, None]
    reference = RBFInterpolator(uniqueX, averagedY, kernel=name, epsilon=1.0, degree=degree)(query)
    with pytest.warns(RuntimeWarning, match="least-squares"):
        model = RBF(kernel=kernelClass()).fit(x, y)
    np.testing.assert_allclose(model.predict(query), reference, rtol=1e-9, atol=1e-9)
    info = model.fitState["linearSolve"]
    assert info["constraintMaxAbsResidual"] < 1e-10
    assert info["relativeTrainingResidual"] > 0.1 if conflict else info["relativeTrainingResidual"] < 1e-10


@pytest.mark.parametrize("scale", [1.0, 1000.0, 10000.0])
def test_singular_cubic_keeps_linear_trend_in_different_units(scale):
    x = np.array([[0.0], [0.1], [0.27], [0.46], [0.65], [0.82], [1.0], [0.27]])
    y = 2 + 3 * x
    query = np.array([[0.07], [0.21], [0.53], [0.91]])
    with pytest.warns(RuntimeWarning, match="singular"):
        model = RBF().fit(x * scale, y)
    np.testing.assert_allclose(model.predict(query * scale), 2 + 3 * query, atol=1e-10, rtol=1e-10)


def test_rank_deficient_trend_is_explicitly_marked():
    x = np.column_stack([np.arange(4.0), np.zeros(4)])
    with pytest.warns(RuntimeWarning, match="unique extrapolation"):
        model = RBF().fit(x, (2 + 3 * x[:, [0]]))
    assert model.fitState["linearSolve"]["deficientTrend"]
    np.testing.assert_allclose(model.predict([[0.5, 0.0], [1.5, 0.0]]), [[3.5], [6.5]], atol=1e-11)


def test_failed_fallback_does_not_reuse_previous_fit(monkeypatch):
    model = RBF().fit(np.arange(4.0)[:, None], np.arange(4.0)[:, None])

    def fail(*args):
        raise np.linalg.LinAlgError("fallback did not converge")

    monkeypatch.setattr(model, "_fitSingularSystem", fail)
    with pytest.raises(np.linalg.LinAlgError, match="did not converge"):
        model.fit(np.array([[0.0], [0.0], [1.0]]), np.array([[0.0], [2.0], [1.0]]))
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict([[0.5]])
