import numpy as np
import pytest

from UQPyL.doe import FFD, FASTSequence, LHS, MorrisSequence, Random, SaltelliSequence, SobolSequence
from UQPyL.problem.problem import Problem


def _make_problem(n_input=3):
    # objective is irrelevant for DOE samplers, but Problem requires nOutput.
    return Problem(nInput=n_input, nOutput=1, ub=[2.0] * n_input, lb=[-1.0] * n_input)


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
    sob = SobolSequence(scramble=True, skipValue=0)
    X1 = sob.sample(problem, nt=16, seed=123)
    X2 = sob.sample(problem, nt=16, seed=123)
    assert X1.shape == (16, 3)
    assert np.all(X1 >= problem.lb - 1e-12)
    assert np.all(X1 <= problem.ub + 1e-12)
    assert np.allclose(X1, X2)


def test_saltelli_sequence_shape_first_order():
    problem = _make_problem(3)
    N = 4
    seq = SaltelliSequence(secondOrder=False, skipValue=0, scramble=True)
    X = seq.sample(problem, nt=N, seed=123)
    assert X.shape == ((problem.nInput + 2) * N, problem.nInput)
    assert np.all(X >= problem.lb - 1e-12)
    assert np.all(X <= problem.ub + 1e-12)


def test_saltelli_sequence_shape_second_order():
    problem = _make_problem(3)
    N = 4
    seq = SaltelliSequence(secondOrder=True, skipValue=0, scramble=True)
    X = seq.sample(problem, nt=N, seed=123)
    assert X.shape == ((2 * problem.nInput + 2) * N, problem.nInput)


def test_saltelli_sequence_skip_value_validation():
    problem = _make_problem(2)
    # skipValue must be positive int power of 2, and N must be >= skipValue
    with pytest.raises(ValueError):
        SaltelliSequence(skipValue=-1).sample(problem, nt=4, seed=1)
    with pytest.raises(ValueError):
        SaltelliSequence(skipValue=3).sample(problem, nt=8, seed=1)  # not power of 2
    with pytest.raises(ValueError):
        SaltelliSequence(skipValue=8).sample(problem, nt=4, seed=1)  # N < skipValue


def test_morris_sequence_shape_and_bounds():
    problem = _make_problem(3)
    mor = MorrisSequence(numLevels=4)
    X = mor.sample(problem, nt=5, seed=123)
    assert X.shape == (5 * (problem.nInput + 1), problem.nInput)
    assert np.all(X >= problem.lb - 1e-12)
    assert np.all(X <= problem.ub + 1e-12)


def test_fast_sequence_shape_bounds_and_min_nt_validation():
    problem = _make_problem(3)
    fast = FASTSequence(M=4)
    with pytest.raises(ValueError):
        fast.sample(problem, nt=4 * fast.M**2, seed=1)  # must be greater than

    X = fast.sample(problem, nt=4 * fast.M**2 + 1, seed=123)
    assert X.shape == ((4 * fast.M**2 + 1) * problem.nInput, problem.nInput)
    assert np.all(X >= problem.lb - 1e-12)
    assert np.all(X <= problem.ub + 1e-12)

