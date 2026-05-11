import numpy as np
import pytest

from UQPyL.problem import ModelEvalContext, ModelProblem, Problem, Space


def test_model_problem_name_default_and_custom():
    p1 = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 4, 3)),
        obs=np.ones((4, 3)),
    )
    assert p1.name == "ModelProblem"

    p2 = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 4, 3)),
        obs=np.ones((4, 3)),
        name="HBVModel",
    )
    assert p2.name == "HBVModel"


def test_model_eval_context_is_public():
    context = ModelEvalContext(sim=np.array([1.0]))

    assert np.allclose(context.sim, [1.0])
    assert not hasattr(context, "raw")


def test_model_problem_simfunc_returns_eval_sim():
    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 2))
        sim[:, 0, 0] = np.sum(X, axis=1)
        sim[:, 0, 1] = np.prod(X, axis=1)
        sim[:, 1, 0] = np.sum(X, axis=1) + 1.0
        sim[:, 1, 1] = np.prod(X, axis=1) + 1.0
        return sim

    p = ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=simf, obs=np.ones((2, 2)))
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    res = p.evaluate(X, target="sim")

    assert res.sim.shape == (2, 2, 2)
    assert np.allclose(res.sim[:, 0, 0], [0.3, 0.7])
    assert np.allclose(res.sim[:, 0, 1], [0.02, 0.12])
    assert p.nObs == 4
    assert p.nOutput == 1
    assert p.simLabels == ["sim_1", "sim_2"]
    assert p.flattenSim(res.sim).shape == (2, 4)
    assert p.flattenObs().shape == (4,)


def test_model_problem_is_problem_and_evaluates_with_context():
    calls = {"sim": 0, "obj": 0, "con": 0}

    def simf(X):
        calls["sim"] += 1
        X = np.atleast_2d(X)
        return np.stack([np.sum(X, axis=1), np.prod(X, axis=1)], axis=1)

    def objf(X, context):
        calls["obj"] += 1
        assert context.sim.shape == (2, 2)
        assert context.obs is None
        return np.sum(context.sim, axis=1, keepdims=True)

    def conf(X, context):
        calls["con"] += 1
        return np.max(context.sim, axis=1, keepdims=True) - 1.0

    p = ModelProblem(
        nInput=2,
        nObj=1,
        nCon=1,
        ub=1.0,
        lb=0.0,
        simFunc=simf,
        objFunc=objf,
        conFunc=conf,
    )
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    res = p.evaluate(X)

    assert isinstance(p, Problem)
    assert np.allclose(res.sim, [[0.3, 0.02], [0.7, 0.12]])
    assert np.allclose(res.objs, [[0.32], [0.82]])
    assert np.allclose(res.cons, [[-0.7], [-0.3]])
    assert calls == {"sim": 1, "obj": 1, "con": 1}


def test_model_problem_target_sim_skips_obj_and_con():
    def fail_obj(X, context):
        raise AssertionError("objFunc should not be called for target='sim'")

    def fail_con(X, context):
        raise AssertionError("conFunc should not be called for target='sim'")

    p = ModelProblem(
        nInput=2,
        nObj=1,
        nCon=1,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.sum(np.atleast_2d(X), axis=1),
        objFunc=fail_obj,
        conFunc=fail_con,
    )

    res = p.evaluate(np.array([[0.1, 0.2], [0.3, 0.4]]), target="sim")
    assert np.allclose(res.sim, [0.3, 0.7])
    assert res.objs is None
    assert res.cons is None


def test_model_problem_objfunc_call_builds_context():
    def simf(X):
        X = np.atleast_2d(X)
        return np.sum(X, axis=1)

    def objf(X, context):
        return (context.sim + 1.0).reshape(-1, 1)

    p = ModelProblem(nInput=2, nObj=1, ub=1.0, lb=0.0, simFunc=simf, objFunc=objf)

    assert np.allclose(p.objFunc(np.array([[0.1, 0.2]])), [[1.3]])


def test_model_problem_rejects_invalid_target():
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 3, 2)),
        obs=np.ones((3, 2)),
    )
    with pytest.raises(ValueError):
        p.evaluate(np.array([[0.1, 0.2]]), target="bad")


def test_model_problem_rejects_missing_callable_configuration():
    with pytest.raises(ValueError, match="simFunc"):
        ModelProblem(nInput=2, ub=1.0, lb=0.0, obs=np.ones((3, 2)))


def test_model_problem_rejects_non_array_sim():
    p = ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=lambda X: [[[1.0, 2.0]]], obs=np.ones((1, 2)))
    with pytest.raises(TypeError, match="np.ndarray"):
        p.evaluate(np.array([[0.1, 0.2]]))


def test_model_problem_accepts_non_3d_sim():
    p = ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=lambda X: np.array([[1.0, 2.0]]), obs=np.ones((1, 2)))
    res = p.evaluate(np.array([[0.1, 0.2]]), target="sim")
    assert np.allclose(res.sim, [[1.0, 2.0]])


def test_model_problem_rejects_wrong_sample_dimension():
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0] + 1, 2, 2)),
        obs=np.ones((2, 2)),
    )
    with pytest.raises(ValueError, match="first dimension"):
        p.evaluate(np.array([[0.1, 0.2], [0.3, 0.4]]))


def test_model_problem_allows_sim_shape_independent_from_obs():
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 3, 2)),
        obs=np.ones((2, 2)),
    )
    res = p.evaluate(np.array([[0.1, 0.2]]), target="sim")
    assert res.sim.shape == (1, 3, 2)


def test_model_problem_rejects_nan_in_sim():
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.array([[[1.0, np.nan]]] * np.atleast_2d(X).shape[0]),
        obs=np.ones((1, 2)),
    )
    with pytest.raises(ValueError, match="NaN"):
        p.evaluate(np.array([[0.1, 0.2]]))


def test_model_problem_supports_custom_space_and_labels():
    space = Space(nInput=2, ub=[1.0, 2.0], lb=[0.0, -1.0], xLabels=["p1", "p2"])
    p = ModelProblem(
        space=space,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 4, 2)),
        obs=np.ones((4, 2)),
        simLabels=["s1", "s2"],
    )
    res = p.evaluate(np.array([[0.1, 0.2]]), target="sim")

    assert p.space is space
    assert p.xLabels == ["p1", "p2"]
    assert p.simLabels == ["s1", "s2"]
    assert res.sim.shape == (1, 4, 2)


def test_model_problem_validates_obs_mask_and_labels():
    with pytest.raises(TypeError, match="obs"):
        ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 2, 2)), obs=[[1.0, 2.0]])

    with pytest.raises(ValueError, match="2D"):
        ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 2, 2)), obs=np.ones(2))

    with pytest.raises(ValueError, match="Mask shape"):
        ModelProblem(
            nInput=2,
            ub=1.0,
            lb=0.0,
            simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 2, 2)),
            obs=np.ones((2, 2)),
            mask=np.ones((4,), dtype=bool),
        )

    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 2, 2)),
        obs=np.ones((2, 2)),
        simLabels=["only_one"],
    )
    assert p.simLabels == ["only_one"]


def test_model_problem_flattens_mask():
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 2, 2)),
        obs=np.ones((2, 2)),
        mask=np.array([[False, True], [True, False]]),
    )
    assert np.array_equal(p.flattenMask(), np.array([False, True, True, False]))
