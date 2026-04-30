import numpy as np
import pytest

from UQPyL.doe import FFD, LHS, Random, Sampler, Sobol
from UQPyL.problem.problem import Problem


def _make_problem(n_input=3):
    # objective is irrelevant for DOE samplers, but Problem requires nObj.
    return Problem(
        nInput=n_input,
        nObj=1,
        ub=[2.0] * n_input,
        lb=[-1.0] * n_input,
        objFunc=lambda X: np.zeros((X.shape[0], 1)),
    )


def test_lhs_classic_sample_shape_and_bounds():
    problem = _make_problem(3)
    X = LHS("classic").sample(problem, nt=10, seed=123)
    assert X.shape == (10, 3)
    assert np.all(X >= problem.lb - 1e-12)
    assert np.all(X <= problem.ub + 1e-12)


@pytest.mark.parametrize("criterion", ["center", "maximin", "center_maximin", "correlation"])
def test_lhs_other_criteria_smoke(criterion):
    problem = _make_problem(3)
    # keep iterations small so CI stays fast; "correlation" criterion can be expensive otherwise
    lhs = LHS(criterion=criterion, iterations=1 if criterion == "correlation" else 3)
    X = lhs.sample(problem, nt=8, seed=123)
    assert X.shape == (8, 3)
    assert np.all(X >= problem.lb - 1e-12)
    assert np.all(X <= problem.ub + 1e-12)


def test_lhs_invalid_criterion_raises():
    problem = _make_problem(2)
    lhs = LHS(criterion="bad")
    with pytest.raises(ValueError):
        lhs.sample(problem, nt=5, seed=1)


def test_lhs_reproducible_given_seed():
    problem = _make_problem(3)
    lhs = LHS("classic")
    X1 = lhs.sample(problem, nt=10, seed=123)
    X2 = lhs.sample(problem, nt=10, seed=123)
    assert np.allclose(X1, X2)


def test_random_sample_shape_and_bounds():
    problem = _make_problem(4)
    X = Random().sample(problem, nt=7, seed=123)
    assert X.shape == (7, 4)
    assert np.all(X >= problem.lb - 1e-12)
    assert np.all(X <= problem.ub + 1e-12)


def test_random_reproducible_given_seed():
    problem = _make_problem(2)
    r = Random()
    X1 = r.sample(problem, nt=20, seed=123)
    X2 = r.sample(problem, nt=20, seed=123)
    assert np.allclose(X1, X2)


def test_ffd_levels_int_produces_cartesian_grid():
    problem = _make_problem(3)
    X = FFD().sample(problem, levels=2, seed=123)
    assert X.shape == (2**3, 3)
    assert np.all(X >= problem.lb - 1e-12)
    assert np.all(X <= problem.ub + 1e-12)


def test_ffd_levels_length_mismatch_raises():
    problem = _make_problem(3)
    with pytest.raises(ValueError):
        FFD().sample(problem, levels=[2, 3], seed=123)  # should be len==nx or scalar


def test_sobol_sequence_shape_bounds_and_reproducible():
    problem = _make_problem(3)
    sob = Sobol(scramble=True, skipValue=0)
    X1 = sob.sample(problem, nt=16, seed=123)
    X2 = sob.sample(problem, nt=16, seed=123)
    assert X1.shape == (16, 3)
    assert np.all(X1 >= problem.lb - 1e-12)
    assert np.all(X1 <= problem.ub + 1e-12)
    assert np.allclose(X1, X2)


def test_sampler_uses_problem_unit_to_space_mapping():
    class ShiftSampler(Sampler):
        def _generate(self, nt: int, nx: int):
            return np.full((nt, nx), 0.5)

    problem = _make_problem(2)
    X = ShiftSampler().sample(problem, nt=3, seed=123)
    expected = np.full((3, 2), 0.5) * (problem.ub - problem.lb) + problem.lb
    assert np.allclose(X, expected)


def test_sampler_invalid_generate_shape_raises():
    class BadShapeSampler(Sampler):
        def _generate(self, nt: int, nx: int):
            return np.zeros((nt, nx + 1))

    problem = _make_problem(3)
    with pytest.raises(ValueError):
        BadShapeSampler().sample(problem, nt=4, seed=123)

