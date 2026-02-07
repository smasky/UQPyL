import numpy as np

from UQPyL.problem.util.non_dominated_sort import NDSort
from UQPyL.problem.util.uniformPoint import uniformPoint


def test_ndsort_basic_two_objectives():
    objs = np.array(
        [
            [0.0, 1.0],  # nondominated
            [1.0, 0.0],  # nondominated
            [1.0, 1.0],  # dominated
        ]
    )
    front, max_front = NDSort(objs)
    assert max_front >= 1
    assert front.shape == (3,)
    assert front[2] > 1


def test_ndsort_all_identical_hits_single_unique_branch():
    objs = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    front, max_front = NDSort(objs)
    assert max_front == 1
    assert np.all(front == 1)


def test_ndsort_with_constraints_penalizes_infeasible():
    objs = np.array([[0.0, 0.0], [0.1, 0.1]])
    cons = np.array([[-1.0], [1.0]])  # second is infeasible
    front, _ = NDSort(objs, popCons=cons)
    # feasible should not be worse than infeasible
    assert front[0] <= front[1]


def test_uniform_point_nbi_and_grid_shapes_and_basic_properties():
    W1, n1 = uniformPoint(10, 3, method="NBI")
    assert W1.shape == (n1, 3)
    assert np.all(W1 >= 0)
    assert np.allclose(W1.sum(axis=1), 1.0, atol=1e-6)

    W2, n2 = uniformPoint(10, 3, method="grid")
    assert W2.shape == (n2, 3)
    assert np.all(W2 >= 0)
    assert np.all(W2 <= 1.0 + 1e-12)


def test_uniform_point_nbi_triggers_secondary_layer_branch():
    # choose N so that H1 < M and H2 > 0 path runs inside `_NBI`
    W, n = uniformPoint(9, 3, method="NBI")
    assert W.shape == (n, 3)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6)


