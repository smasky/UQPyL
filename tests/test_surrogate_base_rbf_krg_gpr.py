import numpy as np
import pytest

from UQPyL.surrogate.base import SurrogateABC
from UQPyL.surrogate.rbf.radial_basis_function import RBF
from UQPyL.surrogate.kriging.kriging import KRG
from UQPyL.surrogate.gp.gaussian_process import GPR
from UQPyL.util.poly import PolyFeature
from UQPyL.util.scaler import MinMaxScaler, StandardScaler


class DummySurrogate(SurrogateABC):
    name = "DummySurrogate"

    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        xTrain, yTrain = self.__check_and_scale__(xTrain, yTrain)
        self.xTrain = xTrain
        self.yTrain = yTrain
        return self

    def predict(self, xPred: np.ndarray):
        xPred = np.atleast_2d(xPred)
        xPred = self.__X_transform__(xPred)
        # simple linear mapping in scaled space
        y = np.sum(xPred, axis=1, keepdims=True)
        return self.__Y_inverse_transform__(y)


def test_surrogateabc_check_and_scale_errors_and_polyfeature_branch():
    s = DummySurrogate(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)), polyFeature=PolyFeature(degree=2))

    with pytest.raises(ValueError):
        s.fit([1, 2, 3], np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        s.fit(np.array([[0.0], [1.0]]), np.array([[1.0], [2.0], [3.0]]))

    X = np.array([[0.0, 0.5], [1.0, 0.25], [0.2, 0.8]])
    Y = np.array([1.0, 2.0, 3.0])
    s.fit(X, Y)
    y_pred = s.predict(np.array([[0.1, 0.2]]))
    assert y_pred.shape == (1, 1)


def test_rbf_fit_predict_and_setkernel_removes_old_kernel_setting():
    X = np.array([[0.0], [0.5], [1.0]])
    Y = np.array([[0.0], [0.25], [1.0]])
    rbf = RBF(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)))
    rbf.fit(X, Y)
    pred = rbf.predict(np.array([[0.2], [0.8]]))
    assert pred.shape == (2, 1)

    # setKernel should work (requires Setting.removeSetting to be correct)
    from UQPyL.surrogate.rbf.kernel.linear_kernel import Linear

    rbf.setKernel(Linear())
    pred2 = rbf.predict(np.array([[0.2]]))
    assert pred2.shape == (1, 1)

    # cover _get_tail_matrix else branch by using a kernel name not handled
    from UQPyL.surrogate.rbf.kernel.gaussian_kernel import Gaussian

    assert rbf._get_tail_matrix(Gaussian(), rbf.xTrain) is None


def test_krg_fit_predict_smoke_with_monkeypatched_boxmin(monkeypatch):
    # Make KRG fit deterministic/fast by patching Boxmin.run to avoid explore loops.
    from UQPyL.surrogate.util.boxmin import Boxmin

    def _fast_run(self, problem, xInit=None):
        if xInit is None:
            xInit = np.random.uniform(problem.lb.ravel(), problem.ub.ravel(), problem.nInput)
        return xInit, float(problem.objFunc(xInit))

    monkeypatch.setattr(Boxmin, "run", _fast_run, raising=True)

    X = np.array([[0.0], [0.5], [1.0], [0.2]])
    Y = np.array([[0.0], [0.25], [1.0], [0.04]])
    krg = KRG(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)), regression="poly1", optimizer="Boxmin")
    krg.fit(X, Y)

    y_pred, mse = krg.predict(np.array([[0.3], [0.7]]), only_value=False)
    assert y_pred.shape == (2, 1)
    assert mse.shape == (2, 1)


def test_gpr_fit_predict_smoke_with_output_std(monkeypatch):
    X = np.array([[0.0], [0.5], [1.0], [0.2]])
    Y = np.array([[0.0], [0.25], [1.0], [0.04]])
    gpr = GPR(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)))
    gpr.fit(X, Y)
    y_mean = gpr.predict(np.array([[0.3], [0.7]]))
    assert y_mean.shape == (2, 1)

    y_mean2, y_std = gpr.predict(np.array([[0.3], [0.7]]), Output_std=True)
    assert y_mean2.shape == (2, 1)
    assert y_std.shape == (2,)

