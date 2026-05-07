import numpy as np
import pytest

from UQPyL.problem import ModelProblem, Space


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
    res = p.evaluate(X)

    assert res.sim.shape == (2, 2, 2)
    assert np.allclose(res.sim[:, 0, 0], [0.3, 0.7])
    assert np.allclose(res.sim[:, 0, 1], [0.02, 0.12])
    assert p.nObs == 4
    assert not hasattr(p, "nOutput")
    assert p.simLabels == ["sim_1", "sim_2"]
    assert p.flattenSim(res.sim).shape == (2, 4)
    assert p.flattenObs().shape == (4,)


def test_model_problem_rejects_invalid_target():
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 3, 2)),
        obs=np.ones((3, 2)),
    )
    with pytest.raises(ValueError):
        p.evaluate(np.array([[0.1, 0.2]]), target="objs")


def test_model_problem_rejects_missing_callable_configuration():
    with pytest.raises(ValueError, match="simFunc"):
        ModelProblem(nInput=2, ub=1.0, lb=0.0, obs=np.ones((3, 2)))
    with pytest.raises(ValueError, match="obs"):
        ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 3, 2)))


def test_model_problem_rejects_non_array_sim():
    p = ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=lambda X: [[[1.0, 2.0]]], obs=np.ones((1, 2)))
    with pytest.raises(TypeError, match="np.ndarray"):
        p.evaluate(np.array([[0.1, 0.2]]))


def test_model_problem_rejects_non_3d_sim():
    p = ModelProblem(nInput=2, ub=1.0, lb=0.0, simFunc=lambda X: np.array([[1.0, 2.0]]), obs=np.ones((1, 2)))
    with pytest.raises(ValueError, match="3D"):
        p.evaluate(np.array([[0.1, 0.2]]))


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


def test_model_problem_rejects_shape_mismatch_with_obs():
    p = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 3, 2)),
        obs=np.ones((2, 2)),
    )
    with pytest.raises(ValueError, match="obs.shape"):
        p.evaluate(np.array([[0.1, 0.2]]))


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
    res = p.evaluate(np.array([[0.1, 0.2]]))

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

    with pytest.raises(ValueError, match="simLabels"):
        ModelProblem(
            nInput=2,
            ub=1.0,
            lb=0.0,
            simFunc=lambda X: np.ones((np.atleast_2d(X).shape[0], 2, 2)),
            obs=np.ones((2, 2)),
            simLabels=["only_one"],
        )


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
