import numpy as np
import pytest

from UQPyL.problem import Eval, ModelEvaluator, ModelProblem, ProblemBase, SimContext, Space


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
    context = SimContext(sims=np.array([1.0]))

    assert np.allclose(context.sims, [1.0])
    assert not hasattr(context, "raw")


def test_model_problem_simfunc_returns_eval_sims():
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
    res = p.evaluate(X, target="sims")

    assert res.sims.shape == (2, 2, 2)
    assert np.allclose(res.sims[:, 0, 0], [0.3, 0.7])
    assert np.allclose(res.sims[:, 0, 1], [0.02, 0.12])
    assert p.nObs == 4
    assert p.nOutput == 1
    assert p.seriesLabels == ["series_1", "series_2"]
    assert p.flattenSim(res.sims).shape == (2, 4)
    assert p.flattenObs().shape == (4,)


def test_model_problem_simulate_returns_sim_context():
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 2, 1)),
        obs=np.ones((2, 1)),
    )
    X = np.array([[0.1, 0.2]])

    context = p.simulate(X)

    assert isinstance(context, SimContext)
    assert np.allclose(context.sims, np.ones((1, 2, 1)))
    assert np.allclose(context.obs, np.ones((2, 1)))
    assert context.mask is None


def test_model_problem_is_problem_base_and_evaluates_with_context():
    calls = {"sim": 0, "obj": 0, "con": 0}

    def simf(X):
        calls["sim"] += 1
        X = np.atleast_2d(X)
        return np.stack([np.sum(X, axis=1), np.prod(X, axis=1)], axis=1)

    def objf(X, context):
        calls["obj"] += 1
        assert context.sims.shape == (2, 2)
        assert context.obs is None
        return np.sum(context.sims, axis=1, keepdims=True)

    def conf(X, context):
        calls["con"] += 1
        return np.max(context.sims, axis=1, keepdims=True) - 1.0

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

    assert isinstance(p, ProblemBase)
    assert np.allclose(res.sims, [[0.3, 0.02], [0.7, 0.12]])
    assert np.allclose(res.objs, [[0.32], [0.82]])
    assert np.allclose(res.cons, [[-0.7], [-0.3]])
    assert calls == {"sim": 1, "obj": 1, "con": 1}


@pytest.mark.parametrize("target", [None, "objs", "cons", "sims"])
def test_model_problem_target_returns_requested_blocks(target):
    X = np.array([[0.2], [0.7]])
    p = ModelProblem(
        nInput=1, nObj=1, nCon=1, lb=0.0, ub=1.0,
        simFunc=lambda X: X * 2,
        objFunc=lambda X, context: context.sims ** 2,
        conFunc=lambda X, context: context.sims - 1,
    )

    res = p.evaluate(X, target=target)

    for block, expected in (("objs", (X * 2) ** 2),
                            ("cons", X * 2 - 1), ("sims", X * 2)):
        if target is None or target == block:
            np.testing.assert_allclose(getattr(res, block), expected)
        else:
            assert getattr(res, block) is None


@pytest.mark.parametrize("target", ["objs", "cons"])
def test_model_problem_validates_simulation_before_filtering(target):
    p = ModelProblem(
        nInput=1, nObj=1, nCon=1, lb=0.0, ub=1.0,
        simFunc=lambda X: np.full_like(X, np.nan),
        objFunc=lambda X, context: X,
        conFunc=lambda X, context: X,
    )
    with pytest.raises(ValueError, match="NaN"):
        p.evaluate(np.array([[0.2]]), target=target)


@pytest.mark.parametrize("target", ["objs", "cons"])
def test_model_problem_custom_evaluate_rejects_unrequested_sims(target):
    class UnfilteredProblem(ModelProblem):
        def evaluate(self, X, target=None):
            return Eval(sims=X, **{target: X})

    p = UnfilteredProblem(
        nInput=1, nObj=1, nCon=1, lb=0.0, ub=1.0,
        simFunc=lambda X: X,
    )
    with pytest.raises(ValueError, match="requires `sims` to be None"):
        p.evaluate(np.array([[0.2]]), target=target)


def test_model_problem_target_sims_skips_obj_and_con():
    def fail_obj(X, context):
        raise AssertionError("objFunc should not be called for target='sims'")

    def fail_con(X, context):
        raise AssertionError("conFunc should not be called for target='sims'")

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

    res = p.evaluate(np.array([[0.1, 0.2], [0.3, 0.4]]), target="sims")
    assert np.allclose(res.sims, [0.3, 0.7])
    assert res.objs is None
    assert res.cons is None


def test_model_problem_supports_custom_evaluator_subclass():
    def simf(X):
        X = np.atleast_2d(X)
        return np.sum(X, axis=1).reshape(-1, 1)

    class PlusOneEvaluator(ModelEvaluator):
        def evaluate(self, X, simContext, target=None):
            sims = simContext.sims
            objs = sims + 1.0
            return Eval(objs=objs, sims=sims, target=target)

    p = ModelProblem(
        nInput=2,
        nObj=1,
        ub=1.0,
        lb=0.0,
        simFunc=simf,
        evaluator=PlusOneEvaluator(),
    )

    res = p.evaluate(np.array([[0.1, 0.2]]))
    assert np.allclose(res.sims, [[0.3]])
    assert np.allclose(res.objs, [[1.3]])


def test_model_problem_rejects_callable_evaluator():
    def simf(X):
        X = np.atleast_2d(X)
        return np.sum(X, axis=1).reshape(-1, 1)

    def evaluateFunc(X, target=None):
        return Eval(objs=np.sum(np.atleast_2d(X), axis=1, keepdims=True))

    with pytest.raises(TypeError, match="ModelEvaluatorBase instance"):
        ModelProblem(
            nInput=2,
            nObj=1,
            ub=1.0,
            lb=0.0,
            simFunc=simf,
            evaluator=evaluateFunc,
        )


def test_model_problem_objfunc_requires_explicit_context():
    def simf(X):
        X = np.atleast_2d(X)
        return np.sum(X, axis=1)

    def objf(X, context):
        return (context.sims + 1.0).reshape(-1, 1)

    p = ModelProblem(nInput=2, nObj=1, ub=1.0, lb=0.0, simFunc=simf, objFunc=objf)

    with pytest.raises(ValueError, match="explicit `context`"):
        p.objFunc(np.array([[0.1, 0.2]]), None)

    context = p.simulate(np.array([[0.1, 0.2]]))
    assert np.allclose(p.objFunc(np.array([[0.1, 0.2]]), context), [[1.3]])


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


def test_model_problem_custom_evaluate_must_return_eval():
    class _BadModelProblem(ModelProblem):
        def evaluate(self, X, target=None):
            return np.zeros((1, 1))

    p = _BadModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 2, 1)),
        objFunc=lambda X, context: np.ones((np.atleast_2d(X).shape[0], 1)),
        obs=np.ones((2, 1)),
    )
    with pytest.raises(TypeError, match="return Eval"):
        p.evaluate(np.array([[0.1, 0.2]]))


def test_model_problem_custom_evaluate_requires_sim():
    class _NoSimModelProblem(ModelProblem):
        def evaluate(self, X, target=None):
            X = self.validate(X)
            return Eval(objs=np.zeros((X.shape[0], 1)))

    p = _NoSimModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 2, 1)),
        objFunc=lambda X, context: np.ones((np.atleast_2d(X).shape[0], 1)),
        obs=np.ones((2, 1)),
    )
    with pytest.raises(ValueError, match="must return `sims`"):
        p.evaluate(np.array([[0.1, 0.2]]))


def test_model_problem_rejects_missing_callable_configuration():
    with pytest.raises(ValueError, match="simFunc"):
        ModelProblem(nInput=2, ub=1.0, lb=0.0, obs=np.ones((3, 2)))


def test_model_problem_rejects_non_array_sim():
    p = ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=lambda X: [[[1.0, 2.0]]], obs=np.ones((1, 2)))
    with pytest.raises(TypeError, match="np.ndarray"):
        p.evaluate(np.array([[0.1, 0.2]]))


def test_model_problem_accepts_non_3d_sim():
    p = ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=lambda X: np.array([[1.0, 2.0]]), obs=np.ones((1, 2)))
    res = p.evaluate(np.array([[0.1, 0.2]]), target="sims")
    assert np.allclose(res.sims, [[1.0, 2.0]])


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
    res = p.evaluate(np.array([[0.1, 0.2]]), target="sims")
    assert res.sims.shape == (1, 3, 2)


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


def test_model_problem_allows_nan_in_masked_sim_positions():
    mask = np.array([[False, True], [False, False]])
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.array([[[1.0, np.nan], [2.0, 3.0]]] * np.atleast_2d(X).shape[0]),
        obs=np.ones((2, 2)),
        mask=mask,
    )

    res = p.evaluate(np.array([[0.1, 0.2]]), target="sims")
    assert res.sims.shape == (1, 2, 2)
    assert np.isnan(res.sims[0, 0, 1])
    assert p.flattenSim(res.sims).shape == (1, 4)


def test_model_problem_rejects_nan_in_unmasked_sim_positions_even_with_mask():
    mask = np.array([[False, True], [False, False]])
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.array([[[np.nan, np.nan], [2.0, 3.0]]] * np.atleast_2d(X).shape[0]),
        obs=np.ones((2, 2)),
        mask=mask,
    )

    with pytest.raises(ValueError, match="outside masked positions"):
        p.evaluate(np.array([[0.1, 0.2]]))


def test_model_problem_supports_custom_space_and_labels():
    space = Space(nInput=2, ub=[1.0, 2.0], lb=[0.0, -1.0], xLabels=["p1", "p2"])
    p = ModelProblem(
        space=space,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 4, 2)),
        obs=np.ones((4, 2)),
        seriesLabels=["s1", "s2"],
    )
    res = p.evaluate(np.array([[0.1, 0.2]]), target="sims")

    assert p.space is space
    assert p.xLabels == ["p1", "p2"]
    assert p.seriesLabels == ["s1", "s2"]
    assert res.sims.shape == (1, 4, 2)


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
        seriesLabels=["only_one"],
    )
    assert p.seriesLabels == ["only_one"]


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
