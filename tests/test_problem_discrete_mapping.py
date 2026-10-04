"""离散辅助映射的小数值、输入类型、副本与正式编码对照。"""

import numpy as np
import pytest

from UQPyL.problem import Problem
from UQPyL.problem.space import Space


@pytest.mark.parametrize("helper", ["map_discrete_vars", "apply_var_type", "transform"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint8, bool, np.float32])
def testDiscreteHelpersPreserveFractionalChoicesAndReadonlyInputs(helper, dtype):
    inputs = np.array([[0], [1]] if dtype is bool else [[0], [1], [2]], dtype=dtype)
    original = inputs.copy()
    inputs.flags.writeable = False
    for choices in [[0.25, 0.75], [-0.1, 0.3]]:
        space = Space(nInput=1, lb=0, ub=2, varType=[2], varSet={0: choices})
        actual = getattr(space, helper)(inputs)
        expected = np.array([choices[0], *[choices[1]] * (len(inputs) - 1)]).reshape(-1, 1)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == inputs.shape and np.all(np.isin(actual, choices))
        assert not np.shares_memory(actual, inputs)
    np.testing.assert_array_equal(inputs, original)


@pytest.mark.parametrize("integerFlag", [False, True])
@pytest.mark.parametrize("discreteFlag", [False, True])
@pytest.mark.parametrize("dtype", [np.int64, float])
def testMixedHelpersRespectFlagsAndPreserveUnrelatedColumns(integerFlag, discreteFlag, dtype):
    inputs = np.array([[0, 1, 0], [2, 2, 1], [4, 3, 2]], dtype=dtype)
    if dtype is float:
        inputs[:, 0] += 0.125
        inputs[:, 1] = [1.6, 2.4, 3.5]
    original = inputs.copy()
    space = Space(nInput=3, lb=[0, 0, 0], ub=[5, 4, 2], varType=[0, 1, 2], varSet={2: [-0.1, 0.3]})
    expected = inputs.astype(float)
    if integerFlag:
        expected[:, 1] = [2, 2, 4] if dtype is float else [1, 2, 3]
    if discreteFlag:
        expected[:, 2] = [-0.1, 0.3, 0.3]
    actual = space.apply_var_type(inputs, IFlag=integerFlag, DFlag=discreteFlag)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(inputs, original)
    assert not np.shares_memory(actual, inputs)


@pytest.mark.parametrize("helper", ["map_discrete_vars", "apply_var_type"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def testProblemDelegatesKeepDecimalDiscreteValues(helper, dtype):
    problem = Problem(nInput=1, nObj=1, lb=0, ub=2, varType=[2], varSet={0: [-0.1, 0.3]}, objFunc=lambda values: values)
    inputs = np.array([[0], [1], [2]], dtype=dtype)
    np.testing.assert_array_equal(getattr(problem, helper)(inputs), [[-0.1], [0.3], [0.3]])


def testFormalMixedUnitDecoderAndRealEvaluationStayCorrect():
    seen = []

    def objective(values):
        seen.append(values.copy())
        return values[:, 2:3]

    problem = Problem(
        nInput=3, nObj=1, lb=[0, 0, 0], ub=[5, 4, 2], varType=[0, 1, 2], varSet={2: [-0.1, 0.3]}, objFunc=objective
    )
    unit = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    expected = np.array([[0, 0, -0.1], [2.5, 2, 0.3], [5, 4, 0.3]])
    decoded = problem.unit_to_space(unit)
    np.testing.assert_array_equal(decoded, expected)
    np.testing.assert_array_equal(problem.unit_to_space(problem.space.space_to_unit(decoded)), expected)
    np.testing.assert_array_equal(problem.evaluate(decoded).objs, expected[:, 2:3])
    np.testing.assert_array_equal(seen[0], expected)
