import sys
from pathlib import Path

import numpy as np
import pytest


# Ensure repo root is on sys.path so `import UQPyL` works without an editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from UQPyL.problem.problem import Problem  # noqa: E402


def test_problem_name_default_and_custom():
    p1 = Problem(nInput=2, nOutput=1, ub=1, lb=0)
    assert p1.name == "Problem"

    p2 = Problem(nInput=2, nOutput=1, ub=1, lb=0, name="MyProb")
    assert p2.name == "MyProb"


def test_problem_bounds_scalar_and_list():
    p = Problem(nInput=3, nOutput=1, ub=5, lb=-2)
    assert p.ub.shape == (1, 3)
    assert p.lb.shape == (1, 3)
    assert np.allclose(p.ub, 5)
    assert np.allclose(p.lb, -2)

    p2 = Problem(nInput=3, nOutput=1, ub=[1, 2, 3], lb=[-1, -2, -3])
    assert p2.ub.shape == (1, 3)
    assert p2.lb.shape == (1, 3)
    assert np.allclose(p2.ub, [[1, 2, 3]])
    assert np.allclose(p2.lb, [[-1, -2, -3]])


def test_problem_opt_type_validation():
    p = Problem(nInput=2, nOutput=1, ub=1, lb=0, optType="min")
    assert p.optType == "min"
    assert p.opt == 1

    p2 = Problem(nInput=2, nOutput=1, ub=1, lb=0, optType="max")
    assert p2.optType == "max"
    assert p2.opt == -1

    with pytest.raises(ValueError):
        Problem(nInput=2, nOutput=2, ub=1, lb=0, optType=["min"])  # len mismatch

    with pytest.raises(ValueError):
        Problem(nInput=2, nOutput=1, ub=1, lb=0, optType="foo")


def test_problem_var_type_and_var_set_validation():
    with pytest.raises(ValueError):
        Problem(nInput=2, nOutput=1, ub=1, lb=0, varType=[0])  # len mismatch

    # discrete variable requires varSet[i] to be list
    with pytest.raises(ValueError):
        Problem(
            nInput=2,
            nOutput=1,
            ub=[1, 1],
            lb=[0, 0],
            varType=[2, 0],
            varSet={0: "not-a-list"},
        )

    p = Problem(
        nInput=2,
        nOutput=1,
        ub=[1, 1],
        lb=[0, 0],
        varType=[2, 0],
        varSet={0: [10, 20, 30]},
    )
    assert p.encoding == "mix"
    assert 0 in p.varSet
    assert p.varSet[0] == [10, 20, 30]


def test_problem_default_obj_and_con_behavior():
    p = Problem(nInput=2, nOutput=1, ub=1, lb=0)
    X = np.array([[0.1, 0.2], [0.3, 0.4]])

    objs = p.objFunc(X)
    assert objs.shape == (2, 1)
    assert np.isinf(objs).all()

    cons = p.conFunc(X)
    assert cons is None

    res = p.evaluate(X)
    assert set(res.keys()) == {"objs", "cons"}
    assert res["objs"].shape == (2, 1)
    assert np.isinf(res["objs"]).all()
    assert res["cons"] is None


def test_problem_evaluate_dispatch_priority():
    # evaluate() should prefer evaluate_ if provided.
    # objFunc()/conFunc() prefer objFunc_/conFunc_ over evaluate_.
    def objf(X):
        X = np.atleast_2d(X)
        return (np.sum(X, axis=1) + 1)[:, None]

    def conf(X):
        X = np.atleast_2d(X)
        return (np.sum(X, axis=1) - 1)[:, None]

    def evalf(X):
        X = np.atleast_2d(X)
        return {
            "objs": (np.sum(X, axis=1) + 100)[:, None],
            "cons": (np.sum(X, axis=1) + 200)[:, None],
        }

    p = Problem(nInput=2, nOutput=1, ub=1, lb=0, objFunc=objf, conFunc=conf, evaluate=evalf)
    X = np.array([[0.1, 0.2], [0.3, 0.4]])

    # evaluate uses evaluate_
    out = p.evaluate(X)
    assert np.allclose(out["objs"], (np.sum(X, axis=1) + 100)[:, None])
    assert np.allclose(out["cons"], (np.sum(X, axis=1) + 200)[:, None])

    # objFunc/conFunc use objFunc_/conFunc_ first
    assert np.allclose(p.objFunc(X), (np.sum(X, axis=1) + 1)[:, None])
    assert np.allclose(p.conFunc(X), (np.sum(X, axis=1) - 1)[:, None])


