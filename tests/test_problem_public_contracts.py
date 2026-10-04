"""Problem public contracts.

Migrated from test_remaining_review.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from UQPyL.problem import Space


# Regression source: test_remaining_review.py::testInvalidVariableTypesAreRejected
@pytest.mark.parametrize(
    "types",
    [[0, 3], [0, -1], [0, 0.5], [0, np.nan], [0, np.inf], [0, 2**32], ["0", "1"], [[0, 1]], [0], 1, [False, True]],
)
def testInvalidVariableTypesAreRejected(types):
    with pytest.raises(ValueError, match="varType"):
        Space(2, ub=10, lb=0, varType=types)


# Regression source: test_remaining_review.py::testValidMixedVariableTypesRoundTrip
def testValidMixedVariableTypesRoundTrip():
    space = Space(3, ub=10, lb=0, varType=[0.0, 1.0, 2.0], varSet={2: [20, 40]})
    values = np.array([[2.0, 8.0, 40.0]])
    np.testing.assert_array_equal(space.unit_to_space(space.space_to_unit(values)), values)


# Regression source: test_remaining_review.py::testProblemStarImportExportsPublicNames
def testProblemStarImportExportsPublicNames():
    import UQPyL.problem as problem

    namespace = {}
    exec("from UQPyL.problem import *", namespace)
    assert all(isinstance(name, str) for name in problem.__all__)
    assert all(namespace[name] is getattr(problem, name) for name in problem.__all__)


# Regression source: test_remaining_review.py::testEvaluationCallsOnlyRequestedCallback
@pytest.mark.parametrize("isModel", [False, True])
@pytest.mark.parametrize("target", [None, "objs", "cons"])
def testEvaluationCallsOnlyRequestedCallback(isModel, target):
    from UQPyL.problem import Problem, ModelProblem

    calls = []

    def objective(X, *context):
        calls.append("objs")
        return X

    def constraint(X, *context):
        calls.append("cons")
        return X - 0.5

    options = dict(nInput=1, nObj=1, nCon=1, lb=0.0, ub=1.0, objFunc=objective, conFunc=constraint)
    problem = ModelProblem(simFunc=lambda X: X, **options) if isModel else Problem(**options)
    result = problem.evaluate([[0.2]], target=target)
    assert calls == (["objs", "cons"] if target is None else [target])
    assert (result.objs is not None) == (target in (None, "objs"))
    assert (result.cons is not None) == (target in (None, "cons"))
