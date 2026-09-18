from types import SimpleNamespace

import numpy as np
import pytest

from UQPyL.optimization.soea import GA
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF, Matern
from UQPyL.surrogate.gp.kernel.c_kernel_ import Constant
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Guass
from UQPyL.surrogate.metric import r_square
from UQPyL.surrogate.rbf import RBF as RBFModel


def lengthAttr(log=True):
    return {"lb": 0.1, "ub": 5.0, "type": "float", "log": log}


def data():
    X = np.random.default_rng(22).uniform(0, 1, (32, 2))
    return X, (np.sin(7 * X[:, 0]) + 0.2 * X[:, 1]).reshape(-1, 1)


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("vectorFirst", [False, True])
def test_apply_uses_all_vector_coordinates_without_shifting_scalar(log, vectorFirst):
    model = GPR(kernel=RBF(heterogeneous=True, length_attr=lengthAttr(log)))
    model.kernel.initialize(2)
    names = ["l", "C"] if vectorFirst else ["C", "l"]
    vector = np.log([0.2, 3.0]) if log else np.array([0.2, 3.0])
    values = np.r_[vector, np.log(1e-7)] if vectorFirst else np.r_[np.log(1e-7), vector]
    original = values.copy()
    model.applyParameterValues(names, values)
    np.testing.assert_allclose(model.setting.get("l"), [0.2, 3.0])
    assert model.setting.get("C") == pytest.approx(1e-7)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("log", [False, True])
def test_kriging_accepts_full_theta_vector(log):
    model = KRG(kernel=Guass(heterogeneous=True, theta_attr=lengthAttr(log)))
    model.kernel.initialize(2)
    model.applyParameterValues(["theta"], np.log([0.2, 3.0]) if log else [0.2, 3.0])
    np.testing.assert_allclose(model.setting.get("theta"), [0.2, 3.0])


@pytest.mark.parametrize("log", [False, True])
def test_vector_values_broadcast_scalar_bounds_consistently(log):
    model = GPR(kernel=RBF(length_scale=[0.2, 3.0], length_attr=lengthAttr(log)))
    model.kernel.initialize(2)
    info, upper, lower = model.setting.getParaInfos(["l", "C"])
    np.testing.assert_array_equal(info["l"], [0, 1])
    np.testing.assert_array_equal(info["C"], [2])
    assert upper.shape == lower.shape == (3,)
    np.testing.assert_allclose(lower[:2], np.log([0.1, 0.1]) if log else [0.1, 0.1])
    np.testing.assert_allclose(upper[:2], np.log([5., 5.]) if log else [5., 5.])


@pytest.mark.parametrize("values", [[0.2], [0.2, 1., 3.]])
def test_wrong_candidate_width_is_rejected_without_numeric_mutation(values):
    model = GPR(kernel=RBF(heterogeneous=True))
    model.kernel.initialize(2)
    original = model.setting.get("l").copy()
    with pytest.raises(ValueError, match="dimension|size|width"):
        model.applyParameterValues(["l"], values)
    np.testing.assert_array_equal(model.setting.get("l"), original)


@pytest.mark.parametrize("coordinate,nu", [(0.125, 0.5), (0.375, 1.5), (0.625, 2.5), (0.875, np.inf)])
def test_numeric_choice_is_not_decoded_twice(coordinate, nu):
    model = GPR(kernel=Matern(heterogeneous=True, optimize_nu=True, length_attr=lengthAttr()))
    model.kernel.initialize(2)
    model.applyParameterValues(["nu", "l"], [coordinate, *np.log([0.2, 3.0])])
    assert model.setting.get("nu") == nu
    np.testing.assert_allclose(model.setting.get("l"), [0.2, 3.0])


def test_fixed_slices_survive_inactive_then_reactivated_kernel_parameter():
    model = GPR(kernel=RBF(heterogeneous=True, length_attr=lengthAttr()))
    model.setKernelChoices([RBF(heterogeneous=True, length_attr=lengthAttr()), Constant()])
    X, Y = data()
    model.fitModel(*model.prepareTrainingData(X, Y))
    names = ["l", "kernel", "C"]
    info, _, _ = model.setting.getParaInfos(names)
    for coordinate in [1.5, 0.5, 1.5, 0.5]:
        model.applyParameterValues(names, [*np.log([0.2, 3.0]), coordinate, np.log(1e-7)], paraInfos=info)
        assert model.setting.get("C") == pytest.approx(1e-7)
        if coordinate == 0.5:
            np.testing.assert_allclose(model.setting.get("l"), [0.2, 3.0])
        else:
            assert "l" not in model.setting.parVal


@pytest.mark.parametrize("modelName", ["GPR", "KRG"])
@pytest.mark.parametrize("seed", [0, 2])
def test_real_ga_joint_tuning_uses_two_dimensions_and_restores_selected_fit(modelName, seed, capsys):
    X, Y = data()
    model = (GPR(kernel=RBF(heterogeneous=True, length_attr=lengthAttr()), C=1e-6, C_attr=None)
             if modelName == "GPR" else KRG(kernel=Guass(heterogeneous=True, theta_attr=lengthAttr())))
    name = "l" if modelName == "GPR" else "theta"
    optimizer = GA(nPop=12, maxFEs=48, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    tuner = AutoTuner(model, optimizer)
    best, score = tuner.optTune(X, Y, paraList=[name], ratio=25, tuneMode="joint", seed=seed)
    assert np.asarray(best).shape == (2,)
    assert np.all(best >= 0.1) and np.all(best <= 5.)
    assert abs(best[0] - best[1]) > 1e-6
    assert np.isfinite(score).all()
    assert "Warning:" not in capsys.readouterr().out
    train, test = (tuner.lastSplit[key] for key in ["train_indices", "test_indices"])
    reference = (GPR(kernel=RBF(length_scale=best, length_attr=None), C=1e-6, C_attr=None)
                 if modelName == "GPR" else KRG(kernel=Guass(theta=best, theta_attr=None)))
    reference.fitModel(*reference.prepareTrainingData(X[train], Y[train]))
    assert float(np.asarray(score).item()) == pytest.approx(r_square(Y[test], reference.predict(X[test])))
    reference.fitModel(*reference.prepareTrainingData(X, Y))
    np.testing.assert_allclose(model.predict(X), reference.predict(X), rtol=1e-7, atol=1e-8)


def test_vector_grid_keeps_each_pair_intact_and_returns_best_real_values():
    X, Y = data()
    vectors = np.array([[0.2, 3.0], [2.0, 0.2], [0.8, 1.5]])
    model = GPR(kernel=RBF(heterogeneous=True, length_attr=lengthAttr()), C=1e-6, C_attr=None)
    tuner = AutoTuner(model)
    best, score = tuner.gridTune(X, Y, paraGrid={"l": np.log(vectors)}, ratio=25, tuneMode="joint", seed=9)
    train, test = (tuner.lastSplit[key] for key in ["train_indices", "test_indices"])
    expectedScores = []
    for vector in vectors:
        reference = GPR(kernel=RBF(length_scale=vector, length_attr=None), C=1e-6, C_attr=None)
        reference.fit(X[train], Y[train])
        expectedScores.append(r_square(Y[test], reference.predict(X[test])))
    np.testing.assert_allclose(best, vectors[np.argmax(expectedScores)])
    assert score == pytest.approx(max(expectedScores))


@pytest.mark.parametrize("entry", ["gridTune", "optTune"])
@pytest.mark.parametrize("winner", ["RBF", "Constant"])
def test_joint_structure_search_initializes_vectors_and_returns_inactive_as_none(entry, winner, monkeypatch):
    X, Y = data()
    model = GPR(kernel=RBF(heterogeneous=True, length_attr=lengthAttr()))
    model.setKernelChoices([RBF(heterogeneous=True, length_attr=lengthAttr()), Constant()])
    names = ["l", "kernel", "C"]
    seen = []
    originalFit = model.fitModel

    def fit(X, Y):
        if model.kernel.displayName == "RBF":
            np.testing.assert_allclose(model.setting.get("l"), [0.2, 3.0])
        assert model.setting.get("C") == pytest.approx(1e-7)
        seen.append(model.kernel.displayName)
        return originalFit(X, Y)

    model.fitModel = fit
    monkeypatch.setattr("UQPyL.surrogate.auto_tuner.r_square", lambda Y, pred: 2. if model.kernel.displayName == winner else 1.)

    class EnumeratingOptimizer:
        def run(self, problem, seed):
            assert problem.nInput == 4
            candidates = np.array([[*np.log([0.2, 3.0]), k, np.log(1e-7)] for k in [1.5, 0.5, 1.5, 0.5]])
            scores = problem.evaluate(candidates).objs
            best = np.argmax(scores[:, 0])
            return SimpleNamespace(bestDecs=candidates[best], bestObjs=scores[best])

    tuner = AutoTuner(model, EnumeratingOptimizer())
    kwargs = ({"paraGrid": {"l": [np.log([0.2, 3.0])], "kernel": [1.5, 0.5, 1.5, 0.5], "C": [np.log(1e-7)]}}
              if entry == "gridTune" else {"paraList": names})
    best, score = getattr(tuner, entry)(X, Y, ratio=25, tuneMode="joint", seed=2, **kwargs)
    assert seen == ["Constant", "RBF", "Constant", "RBF", winner]
    assert best[1].displayName == winner
    if winner == "Constant":
        assert best[0] is None
    else:
        np.testing.assert_allclose(best[0], [0.2, 3.0])
    assert np.asarray(score).item() == 2.


def test_default_grid_preserves_vector_and_does_not_exponentiate_twice():
    X, Y = data()
    model = GPR(kernel=RBF(length_scale=[0.2, 3.0], length_attr=lengthAttr()), C=1e-6, C_attr=None)
    best, score = AutoTuner(model).gridTune(X, Y, owner="kernel", tuneMode="joint", ratio=25, seed=3)
    np.testing.assert_allclose(best, [0.2, 3.0])
    assert np.isfinite(score)


def test_default_grid_preserves_zero_smoothing():
    X, Y = data()
    best, score = AutoTuner(RBFModel()).gridTune(X, Y, owner="model", tuneMode="joint", ratio=25, seed=3)
    assert best == 0.
    assert np.isfinite(score)


@pytest.mark.parametrize("seed", [0, 2])
def test_real_ga_joint_kernel_and_vector_search(seed, capsys):
    X, Y = data()
    model = GPR(kernel=RBF(heterogeneous=True, length_attr=lengthAttr()), C=1e-6, C_attr=None)
    model.setKernelChoices([RBF(heterogeneous=True, length_attr=lengthAttr()),
                            Matern(heterogeneous=True, length_attr=lengthAttr())])
    optimizer = GA(nPop=12, maxFEs=48, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)
    tuner = AutoTuner(model, optimizer)
    (bestVector, bestKernel), score = tuner.optTune(X, Y, paraList=["l", "kernel"], ratio=25,
                                                  tuneMode="joint", seed=seed)
    assert bestVector.shape == (2,)
    assert bestKernel.displayName in {"RBF", "Matern"}
    assert "Warning:" not in capsys.readouterr().out
    reference = GPR(kernel=bestKernel, C=1e-6, C_attr=None)
    train, test = (tuner.lastSplit[key] for key in ["train_indices", "test_indices"])
    reference.fitModel(*reference.prepareTrainingData(X[train], Y[train]))
    assert np.asarray(score).item() == pytest.approx(r_square(Y[test], reference.predict(X[test])))
    reference.fitModel(*reference.prepareTrainingData(X, Y))
    np.testing.assert_allclose(model.predict(X), reference.predict(X), rtol=1e-7, atol=1e-8)


def test_grid_can_apply_parameter_that_becomes_active_after_kernel_switch(monkeypatch):
    X, Y = data()
    model = GPR(kernel=RBF(heterogeneous=True, length_attr=lengthAttr()), C=1e-6, C_attr=None)
    model.setKernelChoices([RBF(heterogeneous=True, length_attr=lengthAttr()),
                            Matern(heterogeneous=True, optimize_nu=True, length_attr=lengthAttr())])
    monkeypatch.setattr("UQPyL.surrogate.auto_tuner.r_square",
                        lambda Y, pred: 2. if model.kernel.displayName == "Matern" else 1.)
    (bestNu, bestVector, bestKernel), score = AutoTuner(model).gridTune(
        X, Y, paraGrid={"nu": [0.625], "l": [np.log([0.2, 3.0])], "kernel": [1.5, 0.5]},
        ratio=25, tuneMode="joint", seed=4)
    assert bestNu == 2.5
    assert bestKernel.displayName == "Matern"
    np.testing.assert_allclose(bestVector, [0.2, 3.0])
    assert score == 2.


@pytest.mark.parametrize("change", ["bounds", "encoding", "dimension"])
def test_opt_tune_rejects_incompatible_shared_parameter_spaces(change):
    X, Y = data()
    attr = lengthAttr(log=change != "encoding")
    if change == "bounds":
        attr["ub"] = 4.0
    model = GPR(kernel=RBF(heterogeneous=True, length_attr=lengthAttr()))
    model.setKernelChoices([RBF(heterogeneous=True, length_attr=lengthAttr()),
                            Matern(heterogeneous=change != "dimension", length_attr=attr)])

    class SelectingOptimizer:
        def run(self, problem, seed):
            problem.evaluate(np.array([[1.5, *np.log([0.2, 3.0])]]))
            pytest.fail("An incompatible search space must be rejected.")

    with pytest.raises(ValueError, match="bounds or encoding|dimension"):
        AutoTuner(model, SelectingOptimizer()).optTune(X, Y, paraList=["kernel", "l"],
                                                      ratio=25, tuneMode="joint", seed=3)


@pytest.mark.parametrize("grid", [{}, {"l": []}])
def test_empty_grids_fail_at_the_input_boundary(grid):
    X, Y = data()
    with pytest.raises(ValueError, match="paraGrid"):
        AutoTuner(GPR()).gridTune(X, Y, paraGrid=grid, ratio=25, tuneMode="joint")


def test_inactive_parameters_are_rejected_when_requested():
    model = GPR(kernel=RBF(heterogeneous=True))
    model.setKernelChoices([RBF(heterogeneous=True), Constant()])
    model.kernel.initialize(2)
    info, _, _ = model.setting.getParaInfos(["kernel", "l"])
    with pytest.raises(KeyError, match="not active"):
        model.applyParameterValues(["kernel", "l"], [1.5, 0., 1.], ignoreInactive=False, paraInfos=info)
