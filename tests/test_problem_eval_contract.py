import numpy as np
import pytest

from UQPyL.problem import Eval, ModelProblem, Problem
from UQPyL.problem.evaluator import Evaluator


def makeProblem(problemClass=Problem, **kwargs):
    config = dict(nInput=2, nObj=1, lb=0, ub=1)
    if issubclass(problemClass, ModelProblem):
        config["simFunc"] = lambda X: X[:, :, None]
    else:
        config["objFunc"] = lambda X: X.sum(axis=1)
    return problemClass(**(config | kwargs))


@pytest.mark.parametrize("baseClass", [Problem, ModelProblem])
@pytest.mark.parametrize("samples", [[0.2, 0.3], [[0.2, 0.3], [0.4, 0.5]]])
def test_overridden_evaluate_receives_normalized_batch(baseClass, samples):
    class CustomProblem(baseClass):
        def evaluate(self, X, target=None):
            assert isinstance(X, np.ndarray) and X.ndim == 2
            return Eval(objs=X.sum(axis=1), target=target)

    result = makeProblem(CustomProblem).evaluate(samples, target="objs")
    np.testing.assert_allclose(result.objs[:, 0], np.atleast_2d(samples).sum(axis=1))


@pytest.mark.parametrize("baseClass", [Problem, ModelProblem])
@pytest.mark.parametrize("invalidInput", [np.ones((2, 1, 2)), [[1, 2, 3]]])
def test_invalid_input_rejected_before_custom_evaluation(baseClass, invalidInput):
    class CustomProblem(baseClass):
        def evaluate(self, X, target=None):
            pytest.fail("Invalid input reached the evaluator")

    with pytest.raises(ValueError):
        makeProblem(CustomProblem).evaluate(invalidInput)


@pytest.mark.parametrize("baseClass", [Problem, ModelProblem])
def test_invalid_target_rejected_before_custom_evaluation(baseClass):
    class CustomProblem(baseClass):
        def evaluate(self, X, target=None):
            pytest.fail("Invalid target reached the evaluator")

    with pytest.raises(ValueError, match="target"):
        makeProblem(CustomProblem).evaluate([0.2, 0.3], target="invalid")


@pytest.mark.parametrize("target", [None, "objs"])
def test_problem_rejects_missing_objectives(target):
    with pytest.raises(ValueError, match="requires `objs`"):
        makeProblem(objFunc=lambda X: None).evaluate([0.2, 0.3], target=target)


@pytest.mark.parametrize("target", [None, "objs"])
def test_custom_evaluator_missing_objectives_is_rejected(target):
    class EmptyEvaluator(Evaluator):
        def evaluate(self, X, target=None):
            return Eval()

    problem = makeProblem(objFunc=None, evaluator=EmptyEvaluator())
    with pytest.raises(ValueError, match="requires `objs`"):
        problem.evaluate([0.2, 0.3], target=target)


@pytest.mark.parametrize("baseClass", [Problem, ModelProblem])
def test_subclass_result_is_validated_after_super_call(baseClass):
    class CustomProblem(baseClass):
        def evaluate(self, X, target=None):
            result = super().evaluate(X, target=target)
            result.objs = np.zeros((len(X), 2))
            return result

    with pytest.raises(ValueError, match="second dimension"):
        makeProblem(CustomProblem).evaluate([0.2, 0.3])


@pytest.mark.parametrize("baseClass", [Problem, ModelProblem])
@pytest.mark.parametrize("target", [None, "cons"])
def test_declared_constraints_must_be_returned(baseClass, target):
    with pytest.raises(ValueError, match="requires `cons`"):
        makeProblem(baseClass, nCon=1).evaluate([0.2, 0.3], target=target)


@pytest.mark.parametrize("baseClass", [Problem, ModelProblem])
def test_unconstrained_cons_request_can_be_empty(baseClass):
    result = makeProblem(baseClass).evaluate([0.2, 0.3], target="cons")
    assert result.objs is None and result.cons is None and result.sims is None


@pytest.mark.parametrize("target", [None, "sims"])
def test_simulation_only_model_remains_supported(target):
    result = makeProblem(ModelProblem).evaluate([0.2, 0.3], target=target)
    np.testing.assert_allclose(result.sims, [[[0.2], [0.3]]])
    assert result.objs is None and result.cons is None


@pytest.mark.parametrize("baseClass", [Problem, ModelProblem])
def test_builtin_evaluate_validates_eval_result_once(baseClass, monkeypatch):
    problem = makeProblem(baseClass)
    validateResult = problem._validate_eval_result
    calls = []

    def trackResult(*args):
        calls.append(args)
        return validateResult(*args)

    monkeypatch.setattr(problem, "_validate_eval_result", trackResult)
    problem.evaluate([0.2, 0.3])
    assert len(calls) == 1


def test_problem_normalizes_input_once(monkeypatch):
    problem = makeProblem()
    validateInput = problem.validate
    calls = []

    def trackInput(X):
        calls.append(X)
        return validateInput(X)

    monkeypatch.setattr(problem, "validate", trackInput)
    problem.evaluate([0.2, 0.3])
    assert len(calls) == 1


def test_scalar_simulation_has_clear_shape_error():
    problem = makeProblem(ModelProblem, simFunc=lambda X: np.array(1.0))
    with pytest.raises(ValueError, match="first dimension"):
        problem.evaluate([0.2, 0.3])


def test_infinite_objective_penalty_remains_supported():
    result = makeProblem(objFunc=lambda X: np.full(X.shape[0], np.inf)).evaluate([0.2, 0.3])
    assert np.isposinf(result.objs[0, 0])
