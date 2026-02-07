import numpy as np
import pytest

from UQPyL.analysis.base import AnalysisABC
from UQPyL.problem.problem import Problem
from UQPyL.util import MinMaxScaler


class DummyAnalysis(AnalysisABC):
    name = "DummyAnalysis"

    def analyze(self, X=None, Y=None):
        return None


def test_analysisabc_scaler_type_validation():
    with pytest.raises(TypeError):
        DummyAnalysis(scalers=(object(), None), verboseFlag=False, logFlag=False, saveFlag=False)
    with pytest.raises(TypeError):
        DummyAnalysis(scalers=(None, object()), verboseFlag=False, logFlag=False, saveFlag=False)


def test_analysisabc_check_y_target_and_index_validation():
    a = DummyAnalysis(scalers=(None, None), verboseFlag=False, logFlag=False, saveFlag=False)

    p = Problem(nInput=2, nOutput=1, ub=1.0, lb=0.0)
    a.setProblem(p)

    X = np.zeros((3, 2))
    # invalid target
    with pytest.raises(ValueError):
        a.check_Y(X, Y=None, target="bad", index="all")

    # index must be list
    with pytest.raises(ValueError):
        a.check_Y(X, Y=np.zeros((3, 1)), target="objFunc", index=0)

    # index out of range triggers "Please check the index you set!"
    with pytest.raises(ValueError):
        a.check_Y(X, Y=np.zeros((3, 1)), target="objFunc", index=[100])


def test_analysisabc_check_and_scale_xy_type_validation_and_reshape():
    a = DummyAnalysis(scalers=(None, None), verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.raises(TypeError):
        a.__check_and_scale_xy__("not-array", np.zeros((3, 1)))
    with pytest.raises(TypeError):
        a.__check_and_scale_xy__(np.zeros((3, 2)), "not-array")

    X = np.zeros((3, 2))
    Y = np.array([1.0, 2.0, 3.0])  # 1d should be reshaped to (n,1)
    X2, Y2 = a.__check_and_scale_xy__(X, Y)
    assert Y2.shape == (3, 1)


def test_analysisabc_evaluate_target_validation():
    a = DummyAnalysis(scalers=(None, None), verboseFlag=False, logFlag=False, saveFlag=False)
    p = Problem(nInput=2, nOutput=1, ub=1.0, lb=0.0)
    a.setProblem(p)
    with pytest.raises(ValueError):
        a.evaluate(np.zeros((2, 2)), target="bad")


def test_analysisabc_scaling_and_reverse_branches_and_setting_helpers():
    # hit xScale/yScale branches, plus Setting.keys/values/getParaValue multi-arg tuple branch
    a = DummyAnalysis(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)), verboseFlag=False, logFlag=False, saveFlag=False)
    p = Problem(nInput=2, nOutput=1, ub=1.0, lb=0.0)
    a.setProblem(p)

    X = np.array([[0.0, 0.5], [1.0, 0.25]])
    Y = np.array([1.0, 2.0])
    Xs, Ys = a.__check_and_scale_xy__(X, Y)
    assert Ys.shape == (2, 1)

    Xr, Yr = a.__reverse_X_Y__(Xs, Ys)
    assert Xr.shape == X.shape
    assert Yr.shape == (2, 1)

    a.setParaValue("a", 1)
    a.setParaValue("b", 2)
    assert set(a.setting.keys()) >= {"a", "b"}
    assert isinstance(a.setting.values(), type({}.values()))
    assert a.getParaValue("a", "b") == (1, 2)


