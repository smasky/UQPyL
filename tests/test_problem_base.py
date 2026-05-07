import numpy as np
import pytest

from UQPyL.problem import ProblemBase
from UQPyL.problem.problem import Problem
from UQPyL.problem.eval import Eval


@ProblemBase.singleFunc
def _sum_obj(x):
    return float(np.sum(x))


def _zero_obj(X):
    X = np.atleast_2d(X)
    return np.zeros((X.shape[0], 1))


def test_problemabc_opt_type_list_branch():
    p = Problem(nInput=2, nObj=2, ub=1.0, lb=0.0, objFunc=lambda X: np.zeros((np.atleast_2d(X).shape[0], 2)), optType=["min", "max"])
    assert p.optType == "min max"
    assert np.allclose(p.opt, np.array([1, -1]))

def test_problemabc_bounds_length_mismatch_raises():
    with pytest.raises(ValueError):
        Problem(nInput=3, nObj=1, ub=[1, 2], lb=[0, 0])


def test_problemabc_conwgt_type_validation():
    with pytest.raises(ValueError):
        Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, conWgt="not-a-list")


def test_problemabc_transform_int_var_rounding():
    p = Problem(nInput=3, nObj=1, ub=1.0, lb=0.0, varType=[1, 0, 1], objFunc=_zero_obj)
    X = np.array([[0.2, 0.3, 1.7], [2.9, 0.1, -3.2]], dtype=float)
    Xt = p.cast_int_vars(X.copy())
    assert np.allclose(Xt[:, [0, 2]], np.round(X[:, [0, 2]]))
    assert np.allclose(Xt[:, 1], X[:, 1])


def test_problemabc_transform_discrete_var_mapping():
    p = Problem(
        nInput=2,
        nObj=1,
        ub=[1.0, 1.0],
        lb=[0.0, 0.0],
        varType=[2, 0],
        varSet={0: [10, 20, 30]},
        objFunc=_zero_obj,
    )
    X = np.array([[0.0, 0.1], [0.34, 0.2], [0.67, 0.3], [1.0, 0.4]], dtype=float)
    Xt = p.map_discrete_vars(X.copy())
    # bins=[0,1/3,2/3,1] -> 0.0->10, 0.34->20, 0.67->30, 1.0->30 (clamped)
    assert np.allclose(Xt[:, 0], [10, 20, 30, 30])


def test_problemabc_transform_to_I_D_respects_flags():
    p = Problem(
        nInput=2,
        nObj=1,
        ub=[1.0, 1.0],
        lb=[0.0, 0.0],
        varType=[2, 1],
        varSet={0: [10, 20]},
        objFunc=_zero_obj,
    )
    X = np.array([[0.9, 1.2]], dtype=float)

    Xt = p.apply_var_type(X.copy(), IFlag=True, DFlag=True)
    assert Xt[0, 0] in (10, 20)
    assert float(Xt[0, 1]).is_integer()

    Xt2 = p.apply_var_type(X.copy(), IFlag=False, DFlag=False)
    assert np.allclose(Xt2, X)


def test_problemabc_single_eval_decorator_stacks_results():
    @ProblemBase.singleEval
    def f(x):
        return Eval(objs=np.array([float(np.sum(x))]), cons=np.array([float(x[0] - 0.5)]))

    X = np.array([[0.0, 1.0], [0.5, 0.5]])
    res = f(X)
    assert isinstance(res, Eval)
    assert res.objs.shape == (2, 1)
    assert res.cons.shape == (2, 1)


def test_problemabc_custom_labels_conwgt_and_evaluate_only_branches():
    # hits xLabels/objLabels else-branches and conWgt list conversion
    p = Problem(
        nInput=2,
        nObj=1,
        ub=np.array([1.0, 2.0]),
        lb=np.array([0.0, -1.0]),
        xLabels=["a", "b"],
        objLabels=["out"],
        conWgt=[1.0],
        evaluate=lambda X: Eval(
            objs=np.ones((np.atleast_2d(X).shape[0], 1)),
            cons=np.zeros((np.atleast_2d(X).shape[0], 1)),
        ),
    )
    assert p.xLabels == ["a", "b"]
    assert p.objLabels == ["out"]
    assert p.conWgt.shape == (1, 1)

    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    # objFunc/conFunc should fall back to _eval_fn if _obj_fn/_con_fn are not provided
    assert p.objFunc(X).shape == (2, 1)
    assert p.conFunc(X).shape == (2, 1)
    assert p.getOptimum() is None


def test_problemabc_set_bounds_invalid_types_raise():
    with pytest.raises(ValueError):
        Problem(nInput=2, nObj=1, ub=(1.0, 2.0), lb=[0.0, 0.0])
    with pytest.raises(ValueError):
        Problem(nInput=2, nObj=1, ub=[1.0, 2.0], lb=(0.0, 0.0))


def test_problemabc_check_opttype_invalid_list_and_type():
    with pytest.raises(ValueError):
        Problem(nInput=2, nObj=2, ub=1.0, lb=0.0, optType=["min", "bad"])
    with pytest.raises(ValueError):
        Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, optType=123)


def test_problemabc_transform_unit_x_hits_mix_branch():
    p = Problem(
        nInput=2,
        nObj=1,
        ub=[1.0, 1.0],
        lb=[0.0, 0.0],
        varType=[2, 1],
        varSet={0: [10, 20]},
        objFunc=_zero_obj,
    )
    X = np.array([[0.0, 0.2], [1.0, 0.8]], dtype=float)
    Xt = p.unit_to_space(X.copy(), IFlag=True, DFlag=True)
    assert Xt.shape == X.shape


def test_problemabc_transform_unit_x_uses_direct_unit_scaling():
    p = Problem(nInput=2, nObj=1, ub=[10.0, 20.0], lb=[0.0, -10.0], objFunc=_zero_obj)
    X = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=float)
    Xt = p.unit_to_space(X.copy(), IFlag=False, DFlag=False)
    assert np.allclose(Xt, [[2.5, -2.5], [7.5, 12.5]])

    Xt2 = p.space.unit_to_space(X.copy(), IFlag=False, DFlag=False)
    assert np.allclose(Xt2, Xt)


def test_problem_space_missing_discrete_varset_raises_cleanly():
    with pytest.raises(ValueError):
        Problem(
            nInput=2,
            nObj=1,
            ub=[1.0, 1.0],
            lb=[0.0, 0.0],
            varType=[2, 0],
            varSet={},
        )


def test_problem_space_validate_rejects_wrong_column_count():
    p = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, objFunc=_zero_obj)
    with pytest.raises(ValueError):
        p.space.validate(np.array([[0.1, 0.2, 0.3]]))


def test_problem_space_new_naming_methods_work():
    p = Problem(
        nInput=2,
        nObj=1,
        ub=[1.0, 1.0],
        lb=[0.0, 0.0],
        varType=[2, 1],
        varSet={0: [10, 20]},
        objFunc=_zero_obj,
    )
    X = np.array([[0.9, 1.2]], dtype=float)

    Xt_int = p.space.cast_int_vars(X.copy())
    assert float(Xt_int[0, 1]).is_integer()

    Xt_disc = p.space.map_discrete_vars(X.copy())
    assert Xt_disc[0, 0] in (10, 20)

    Xt_cast = p.space.apply_var_type(X.copy())
    assert Xt_cast[0, 0] in (10, 20)
    assert float(Xt_cast[0, 1]).is_integer()
