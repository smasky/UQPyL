"""Independent construction invariants promoted from the DoE audit."""

import numpy as np
import pytest

from UQPyL.doe import FASTDesign, MorrisDesign, SaltelliDesign
from UQPyL.problem import Problem

pytestmark = pytest.mark.numerical


def makeProblem(dimension):
    return Problem(nInput=dimension, nObj=1, lb=0.0, ub=1.0, objFunc=lambda x: x[:, :1])


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("secondOrder", [False, True])
def testSaltelliRowsReplaceExactlyOneCoordinate(dimension, secondOrder):
    points, meta = SaltelliDesign(secondOrder=secondOrder).sampleWithMeta(
        makeProblem(dimension), 16, seed=17, output="unit"
    )
    blockSize = (2 * dimension if secondOrder else dimension) + 2
    assert meta["blockSize"] == blockSize
    assert points.shape == (16 * blockSize, dimension)
    blocks = points.reshape(16, blockSize, dimension)
    for axis in range(dimension):
        expected = blocks[:, 0].copy()
        expected[:, axis] = blocks[:, -1, axis]
        np.testing.assert_array_equal(blocks[:, axis + 1], expected)
        if secondOrder:
            expected = blocks[:, -1].copy()
            expected[:, axis] = blocks[:, 0, axis]
            np.testing.assert_array_equal(blocks[:, dimension + axis + 1], expected)


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("levels", [4, 8])
def testMorrisTrajectoryVisitsEveryAxisOnceOnLevelGrid(dimension, levels):
    points = MorrisDesign(levels).sample(makeProblem(dimension), 8, seed=17, output="unit")
    assert points.shape == (8 * (dimension + 1), dimension)
    assert np.all((points >= 0) & (points <= 1))
    differences = np.diff(points.reshape(8, dimension + 1, dimension), axis=1)
    active = np.abs(differences) > 1e-12
    np.testing.assert_array_equal(active.sum(axis=1), np.ones((8, dimension)))
    np.testing.assert_array_equal(active.sum(axis=2), np.ones((8, dimension)))
    np.testing.assert_allclose(np.abs(differences[active]), levels / (2 * (levels - 1)), atol=1e-14)
    np.testing.assert_allclose(points * (levels - 1), np.round(points * (levels - 1)), atol=1e-14)


@pytest.mark.parametrize("dimension", [1, 3])
def testFastTrajectoryMatchesIndependentTriangularWave(dimension):
    count, harmonics, seed = 1025, 4, 17
    points = FASTDesign(M=harmonics).sample(makeProblem(dimension), count, seed=seed, output="unit")
    assert points.shape == (dimension * count, dimension)
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=dimension)
    high = (count - 1) // (2 * harmonics)
    low = np.floor(np.linspace(1, high // (2 * harmonics), dimension - 1))
    for focus in range(dimension):
        frequencies = np.insert(low, focus, high)
        angle = np.arange(count)[:, None] * (2 * np.pi / count) * frequencies + phases[focus]
        # Triangle-wave identity, without the implementation's arcsin(sin()).
        expected = 1 - np.abs(((angle + np.pi / 2) % (2 * np.pi)) / np.pi - 1)
        np.testing.assert_allclose(points[focus * count : (focus + 1) * count], expected, atol=2e-12, rtol=0)
