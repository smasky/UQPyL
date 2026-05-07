import numpy as np
import pytest

from UQPyL.analysis.base import AnalysisABC
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


class DummyAnalysis(AnalysisABC):
    name = "DummyAnalysis"

    def _analyzeCore(self, problem=None, X=None, Y=None):
        return None


@ProblemABC.singleFunc
def _zero_obj(x):
    return 0.0


def test_analysisabc_check_y_target_and_index_validation():
    a = DummyAnalysis(verboseFlag=False, logFlag=False, saveFlag=False)

    p = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, objFunc=_zero_obj)
    a.setProblem(p)

    X = np.zeros((3, 2))
    # invalid target
    with pytest.raises(ValueError):
        a.check_Y(X, Y=None, target="bad", index="all")

    # scalar index is now accepted
    Y0 = a.check_Y(X, Y=np.zeros((3, 1)), target="objs", index=0)
    assert Y0.shape == (3, 1)

    # index out of range triggers "Please check the index you set!"
    with pytest.raises(ValueError):
        a.check_Y(X, Y=np.zeros((3, 1)), target="objs", index=[100])


def test_analysisabc_check_xy_type_validation_and_reshape():
    a = DummyAnalysis(verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.raises(TypeError):
        a.__check_X_Y__("not-array", np.zeros((3, 1)))
    with pytest.raises(TypeError):
        a.__check_X_Y__(np.zeros((3, 2)), "not-array")

    X = np.zeros((3, 2))
    Y = np.array([1.0, 2.0, 3.0])  # 1d should be reshaped to (n,1)
    X2, Y2 = a.__check_X_Y__(X, Y)
    assert X2.shape == (3, 2)
    assert Y2.shape == (3, 1)


def test_analysisabc_evaluate_target_validation():
    a = DummyAnalysis(verboseFlag=False, logFlag=False, saveFlag=False)
    p = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, objFunc=_zero_obj)
    a.setProblem(p)
    with pytest.raises(ValueError):
        a.evaluate(np.zeros((2, 2)), target="bad")


def test_analysisabc_setting_helpers():
    a = DummyAnalysis(verboseFlag=False, logFlag=False, saveFlag=False)
    a.set("a", 1)
    a.set("b", 2)
    assert set(a.setting.keys()) >= {"a", "b"}
    assert isinstance(a.setting.values(), type({}.values()))
    assert a.get("a", "b") == (1, 2)


