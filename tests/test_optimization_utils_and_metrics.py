import numpy as np
import pytest

from UQPyL.optimization.metric import GD, IGD, HV
from UQPyL.optimization.core import NDSort, crowdingDist, tourSelect, uniformPoint


def test_ndsort_basic_and_with_constraints_does_not_crash():
    popObjs = np.array(
        [
            [1.0, 1.0],
            [0.5, 2.0],
            [2.0, 0.5],
        ]
    )
    frontNo, maxF = NDSort(popObjs.copy(), popCons=None, nSort=None)
    assert frontNo.shape == (3,)
    assert maxF >= 1

    # constraint path: infeasible points get penalized
    popCons = np.array([[0.0], [1.0], [-1.0]])
    frontNo2, maxF2 = NDSort(popObjs.copy(), popCons=popCons, nSort=3)
    assert frontNo2.shape == (3,)
    assert maxF2 >= 1


def test_crowding_distance_branches():
    popObjs = np.array([[1.0, 1.0], [1.0, 1.0]])
    # all inf frontNo -> returns zeros
    cd0 = crowdingDist(popObjs, frontNo=np.array([np.inf, np.inf]))
    assert np.allclose(cd0, 0.0)

    # n_f == 1 -> inf
    cd1 = crowdingDist(np.array([[1.0, 2.0]]), frontNo=np.array([1]))
    assert np.isinf(cd1[0])

    # n_f == 2 -> both inf
    cd2 = crowdingDist(np.array([[1.0, 2.0], [2.0, 1.0]]), frontNo=np.array([1, 1]))
    assert np.all(np.isinf(cd2))

    # denom==0 across all dims -> uses boundary inf and continue
    popObjs3 = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    cd3 = crowdingDist(popObjs3, frontNo=np.array([1, 1, 1]))
    assert np.isinf(cd3[0]) and np.isinf(cd3[-1])


def test_tournament_selection_errors_and_empty():
    with pytest.raises(ValueError):
        tourSelect(2, 3)
    out = tourSelect(2, 3, np.zeros((0, 1)))
    assert out.shape == (0,)


def test_tournament_selection_smoke_reproducible_shape():
    np.random.seed(123)
    f1 = np.array([[3.0], [2.0], [1.0], [0.0]])
    f2 = np.array([[0.0], [0.0], [1.0], [2.0]])
    idx = tourSelect(2, 5, f1, f2)
    assert idx.shape == (5,)
    assert np.all((0 <= idx) & (idx < 4))


def test_uniform_point_grid_and_nbi_shapes_and_sums():
    Wg, Ng = uniformPoint(10, 3, method="grid")
    assert Wg.shape[1] == 3
    assert Ng == Wg.shape[0]

    W, N = uniformPoint(10, 3, method="NBI")
    assert W.shape[1] == 3
    assert N == W.shape[0]
    # after re-normalization, weights sum to 1
    assert np.allclose(W.sum(axis=1), 1.0)


def test_metrics_gd_igd_hv_smoke_and_hv_high_dim_branch():
    optimum = np.array([[0.0, 0.0], [1.0, 1.0]])
    pop = np.array([[0.0, 0.0], [0.5, 0.5]])
    assert GD(pop, optimum) >= 0
    assert IGD(pop, optimum) >= 0
    assert HV(pop, refPoint=np.array([1.0, 1.0]), normalize=False) >= 0

    # cover m>=4 Monte Carlo branch quickly via small nSamples
    pop4 = np.array([[0.2, 0.2, 0.2, 0.2], [0.4, 0.1, 0.3, 0.2]])
    hv = HV(pop4, refPoint=np.array([1.0, 1.0, 1.0, 1.0]), normalize=False, nSamples=2000)
    assert hv >= 0


def test_gd_and_igd_match_known_asymmetric_values():
    optimum = np.array([[0.0, 0.0], [1.0, 0.0]])
    pop = np.array([[0.0, 0.0]])

    # GD: from each population point to the nearest optimum point.
    assert GD(pop, optimum) == pytest.approx(0.0)

    # IGD: from each optimum point to the nearest population point.
    assert IGD(pop, optimum) == pytest.approx(0.5)


def test_gd_and_igd_are_zero_on_exact_match():
    optimum = np.array([[0.0, 0.0], [1.0, 1.0]])
    pop = optimum.copy()

    assert GD(pop, optimum) == pytest.approx(0.0)
    assert IGD(pop, optimum) == pytest.approx(0.0)


def test_gd_and_igd_raise_on_empty_input():
    with pytest.raises(ValueError):
        GD(np.empty((0, 2)), np.array([[0.0, 0.0]]))

    with pytest.raises(ValueError):
        IGD(np.array([[0.0, 0.0]]), np.empty((0, 2)))


def test_hv_returns_zero_for_points_outside_reference_box():
    pop = np.array([[2.0, 0.0]])
    hv = HV(pop, refPoint=np.array([1.0, 1.0]), normalize=False)
    assert hv == pytest.approx(0.0)


def test_hv_matches_known_2d_union_area():
    pop = np.array([[0.2, 0.8], [0.8, 0.2]])
    hv = HV(pop, refPoint=np.array([1.0, 1.0]), normalize=False)
    assert hv == pytest.approx(0.28)


def test_hv_matches_known_3d_single_point_volume():
    pop = np.array([[0.5, 0.5, 0.5]])
    hv = HV(pop, refPoint=np.array([1.0, 1.0, 1.0]), normalize=False)
    assert hv == pytest.approx(0.125)


def test_hv_matches_known_3d_union_volume():
    pop = np.array([
        [0.2, 0.7, 0.9],
        [0.6, 0.3, 0.5],
    ])
    hv = HV(pop, refPoint=np.array([1.0, 1.0, 1.0]), normalize=False)
    assert hv == pytest.approx(0.152)


def test_hv_normalize_preserves_value_inside_unit_box():
    pop = np.array([[0.2, 0.8], [0.8, 0.2]])
    ref = np.array([1.0, 1.0])
    assert HV(pop, refPoint=ref, normalize=True) == pytest.approx(0.28)


def test_hv_helpers_head_tail_insert_add_slice_branches():
    # cover helper functions in hv.py that are otherwise hard to hit
    from UQPyL.optimization.metric.hv import add, head, insert, slice, tail

    assert head([]) == []
    assert tail([]) == []
    assert tail([np.array([1.0, 1.0])]) == []

    # add(): merge same ql branch
    S = [(1.0, [np.array([0.1, 0.2])])]
    S2 = add([2.0, [np.array([0.1, 0.2])]], S)
    assert S2[0][0] == 3.0

    # insert(): walk both flags and keep entries
    pl = [np.array([0.2, 0.2]), np.array([0.3, 0.1])]
    out = insert(np.array([0.25, 0.15]), 1, pl)
    assert len(out) >= 2

    # slice(): empty pl branch after head/tail
    pl2 = [np.array([0.1, 0.9]), np.array([0.2, 0.8])]
    S = slice(pl2, 0, refPoint=np.array([1.0, 1.0]))
    assert isinstance(S, list)

