import sys
from pathlib import Path

import numpy as np
import pytest


# Ensure repo root is on sys.path so `import UQPyL` works without an editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from UQPyL.problem.evaluator import Evaluator  # noqa: E402
from UQPyL.problem.problem import Problem  # noqa: E402
from UQPyL.problem.eval import Eval  # noqa: E402
from UQPyL.problem.space import Space  # noqa: E402


def _zero_obj(X):
    X = np.atleast_2d(X)
    return np.zeros((X.shape[0], 1))


class _EvalOnlyProblem(Problem):
    def evaluate(self, X, target=None):
        if target not in (None, "objs", "cons"):
            raise ValueError("The target must be None, 'objs' or 'cons'.")

        X = self.validate(X)
        res = Eval(
            objs=(np.sum(X, axis=1) + 100)[:, None],
            cons=(np.sum(X, axis=1) + 200)[:, None],
            target=target,
        )
        return res

    def objFunc(self, X):
        return self.evaluate(X, target="objs").objs

    def conFunc(self, X):
        return self.evaluate(X, target="cons").cons


def test_problem_name_default_and_custom():
    p1 = Problem(nInput=2, nObj=1, ub=1, lb=0, objFunc=_zero_obj)
    assert p1.name == "Problem"

    p2 = Problem(nInput=2, nObj=1, ub=1, lb=0, objFunc=_zero_obj, name="MyProb")
    assert p2.name == "MyProb"


def test_problem_bounds_scalar_and_list():
    p = Problem(nInput=3, nObj=1, ub=5, lb=-2, objFunc=_zero_obj)
    assert p.ub.shape == (1, 3)
    assert p.lb.shape == (1, 3)
    assert np.allclose(p.ub, 5)
    assert np.allclose(p.lb, -2)

    p2 = Problem(nInput=3, nObj=1, ub=[1, 2, 3], lb=[-1, -2, -3], objFunc=_zero_obj)
    assert p2.ub.shape == (1, 3)
    assert p2.lb.shape == (1, 3)
    assert np.allclose(p2.ub, [[1, 2, 3]])
    assert np.allclose(p2.lb, [[-1, -2, -3]])


def test_problem_opt_type_validation():
    p = Problem(nInput=2, nObj=1, ub=1, lb=0, objFunc=_zero_obj, optType="min")
    assert p.optType == "min"
    assert p.opt == 1
    assert p.nObj == 1

    p2 = Problem(nInput=2, nObj=1, ub=1, lb=0, objFunc=_zero_obj, optType="max")
    assert p2.optType == "max"
    assert p2.opt == -1

    with pytest.raises(ValueError):
        Problem(nInput=2, nObj=2, ub=1, lb=0, objFunc=lambda X: np.zeros((np.atleast_2d(X).shape[0], 2)), optType=["min"])  # len mismatch

    with pytest.raises(ValueError):
        Problem(nInput=2, nObj=1, ub=1, lb=0, objFunc=_zero_obj, optType="foo")


def test_problem_var_type_and_var_set_validation():
    with pytest.raises(ValueError):
        Problem(nInput=2, nObj=1, ub=1, lb=0, varType=[0])  # len mismatch

    # discrete variable requires varSet[i] to be list
    with pytest.raises(ValueError):
        Problem(
            nInput=2,
            nObj=1,
            ub=[1, 1],
            lb=[0, 0],
            varType=[2, 0],
            varSet={0: "not-a-list"},
        )

    p = Problem(
        nInput=2,
        nObj=1,
        ub=[1, 1],
        lb=[0, 0],
        varType=[2, 0],
        varSet={0: [10, 20, 30]},
        objFunc=_zero_obj,
    )
    assert 0 in p.varSet
    assert p.varSet[0] == [10, 20, 30]
    assert p.idxD.tolist() == [0]


def test_problem_evaluate_only_behavior():
    p = _EvalOnlyProblem(nInput=2, nObj=1, nCon=1, ub=1, lb=0, objFunc=_zero_obj)
    X = np.array([[0.1, 0.2], [0.3, 0.4]])

    out = p.evaluate(X)
    assert np.allclose(out.objs, (np.sum(X, axis=1) + 100)[:, None])
    assert np.allclose(out.cons, (np.sum(X, axis=1) + 200)[:, None])
    assert np.allclose(p.objFunc(X), (np.sum(X, axis=1) + 100)[:, None])
    assert np.allclose(p.conFunc(X), (np.sum(X, axis=1) + 200)[:, None])


def test_problem_evaluate_target_objs_and_cons_split():
    def objf(X):
        X = np.atleast_2d(X)
        return (np.sum(X, axis=1) + 1)[:, None]

    def conf(X):
        X = np.atleast_2d(X)
        return (np.sum(X, axis=1) - 1)[:, None]

    p = Problem(nInput=2, nObj=1, ub=1, lb=0, objFunc=objf, conFunc=conf, nCon=1)
    X = np.array([[0.1, 0.2], [0.3, 0.4]])

    obj_res = p.evaluate(X, target="objs")
    assert isinstance(obj_res, Eval)
    assert np.allclose(obj_res.objs, (np.sum(X, axis=1) + 1)[:, None])
    assert obj_res.cons is None

    con_res = p.evaluate(X, target="cons")
    assert isinstance(con_res, Eval)
    assert con_res.objs is None
    assert np.allclose(con_res.cons, (np.sum(X, axis=1) - 1)[:, None])


def test_problem_evaluate_invalid_target_raises():
    p = Problem(nInput=2, nObj=1, ub=1, lb=0, objFunc=_zero_obj)
    with pytest.raises(ValueError):
        p.evaluate(np.array([[0.1, 0.2]]), target="bad")


def test_problem_custom_evaluate_must_return_eval():
    class _BadProblem(Problem):
        def evaluate(self, X, target=None):
            return np.zeros((1, 1))

    p = _BadProblem(nInput=2, nObj=1, ub=1, lb=0, objFunc=_zero_obj)
    with pytest.raises(TypeError, match="return Eval"):
        p.evaluate(np.array([[0.1, 0.2]]))


def test_problem_custom_evaluate_shape_is_validated():
    class _BadShapeProblem(Problem):
        def evaluate(self, X, target=None):
            X = self.validate(X)
            return Eval(objs=np.zeros((X.shape[0], 2)))

    p = _BadShapeProblem(nInput=2, nObj=1, ub=1, lb=0, objFunc=_zero_obj)
    with pytest.raises(ValueError, match="second dimension"):
        p.evaluate(np.array([[0.1, 0.2]]))


def test_problem_supports_custom_space_and_transparent_fields():
    space = Space(
        nInput=2,
        ub=[1.0, 2.0],
        lb=[0.0, -1.0],
        xLabels=["a", "b"],
    )

    p = Problem(space=space, nObj=1, objFunc=_zero_obj, name="P")
    assert p.space is space
    assert p.nInput == 2
    assert p.nObj == 1
    assert p.nCon == 0
    assert p.xLabels == ["a", "b"]
    assert np.allclose(p.ub, [[1.0, 2.0]])
    assert np.allclose(p.lb, [[0.0, -1.0]])


def test_problem_supports_component_style_evaluator():
    evaluator = Evaluator(
        objFunc=lambda X: np.sum(np.atleast_2d(X), axis=1, keepdims=True),
        conFunc=lambda X: (np.atleast_2d(X)[:, 0] - 1.0).reshape(-1, 1),
    )
    p = Problem(
        nInput=2,
        nObj=1,
        nCon=1,
        ub=1.0,
        lb=0.0,
        evaluator=evaluator,
    )

    res = p.evaluate(np.array([[0.2, 0.3]]))
    assert np.allclose(res.objs, [[0.5]])
    assert np.allclose(res.cons, [[-0.8]])


def test_problem_supports_custom_evaluator_subclass():
    class SumEvaluator(Evaluator):
        def evaluate(self, X, target=None):
            X = np.atleast_2d(X)
            objs = np.sum(X, axis=1, keepdims=True)
            cons = (X[:, 0] - 1.0).reshape(-1, 1)
            return Eval(objs=objs, cons=cons, target=target)

    p = Problem(
        nInput=2,
        nObj=1,
        nCon=1,
        ub=1.0,
        lb=0.0,
        evaluator=SumEvaluator(),
    )

    res = p.evaluate(np.array([[0.2, 0.3]]))
    assert np.allclose(res.objs, [[0.5]])
    assert np.allclose(res.cons, [[-0.8]])


def test_problem_rejects_callable_evaluator():
    def evaluateFunc(X, target=None):
        X = np.atleast_2d(X)
        objs = np.sum(X, axis=1, keepdims=True)
        cons = (X[:, 0] - 1.0).reshape(-1, 1)
        return Eval(objs=objs, cons=cons)

    with pytest.raises(TypeError, match="EvaluatorBase instance"):
        Problem(
            nInput=2,
            nObj=1,
            nCon=1,
            ub=1.0,
            lb=0.0,
            evaluator=evaluateFunc,
        )


def test_problem_new_nobj_ncon_and_label_fields():
    p = Problem(
        nInput=2,
        nObj=2,
        nCon=1,
        ub=1.0,
        lb=0.0,
        objFunc=lambda X: np.zeros((np.atleast_2d(X).shape[0], 2)),
        objLabels=["f1", "f2"],
        conLabels=["g1"],
    )
    assert p.nObj == 2
    assert p.nCon == 1
    assert p.objLabels == ["f1", "f2"]
    assert p.conLabels == ["g1"]


def test_problem_rejects_confunc_without_objfunc():
    with pytest.raises(ValueError, match="conFunc"):
        Problem(
            nInput=2,
            nObj=1,
            ub=1.0,
            lb=0.0,
            conFunc=lambda X: np.zeros((np.atleast_2d(X).shape[0], 1)),
        )


def test_problem_rejects_missing_callable_configuration():
    with pytest.raises(ValueError, match="requires"):
        Problem(
            nInput=2,
            nObj=1,
            ub=1.0,
            lb=0.0,
        )


def test_problem_objfunc_missing_raises():
    class _ObjMissingProblem(Problem):
        def __init__(self):
            super().__init__(nInput=2, nObj=1, ub=1.0, lb=0.0)

        def _validate_callable_config(self, objFunc, conFunc, evaluator):
            return None

    p = _ObjMissingProblem()
    with pytest.raises(ValueError, match="objFunc"):
        p.objFunc(np.array([[0.1, 0.2]]))
