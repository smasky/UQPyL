import numpy as np
import pytest

from UQPyL.optimization.metric import HV


@pytest.mark.parametrize("batchSize", [1, 37, 4096, 10000])
@pytest.mark.parametrize("nPoints,nObjectives", [(1, 4), (39, 6), (513, 4)])
def test_batches_match_unbatched_estimate_and_rng_state(batchSize, nPoints, nObjectives):
    points = np.random.default_rng(18).uniform(-0.5, 1, (nPoints, nObjectives))
    ref = np.full(nObjectives, 1.5)
    count = 4217
    expectedRng = np.random.default_rng(27)
    actualRng = np.random.default_rng(27)
    lower = points.min(axis=0)
    samples = expectedRng.uniform(lower, ref, (count, nObjectives))
    dominated = np.any(np.all(points <= samples[:, None], axis=2), axis=1)
    expected = np.sum(dominated) / count * np.prod(ref - lower)
    actual = HV(points, ref, normalize=False, nSamples=count,
                rng=actualRng, batchSize=batchSize)
    assert actual == expected
    np.testing.assert_array_equal(actualRng.random(10), expectedRng.random(10))


def test_normalization_and_outside_reference_points_are_batch_independent():
    points = np.random.default_rng(10).uniform(-2, 4, (300, 5))
    ref = np.full(5, 3.0)
    small = HV(points, ref, nSamples=1031, batchSize=31, rng=np.random.default_rng(9))
    large = HV(points, ref, nSamples=1031, batchSize=2000, rng=np.random.default_rng(9))
    assert small == large


def test_rng_never_receives_more_than_batch_size():
    class TrackingRng:
        def __init__(self):
            self.rng = np.random.default_rng(3)
            self.sizes = []

        def uniform(self, lower, upper, size):
            self.sizes.append(size)
            return self.rng.uniform(lower, upper, size)

    rng = TrackingRng()
    assert HV(np.zeros((1, 4)), np.ones(4), normalize=False,
              nSamples=103, batchSize=32, rng=rng) == 1.0
    assert rng.sizes == [(32, 4), (32, 4), (32, 4), (7, 4)]


@pytest.mark.parametrize("batchSize", [0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected(batchSize):
    with pytest.raises(ValueError, match="batchSize"):
        HV(np.zeros((1, 4)), np.ones(4), nSamples=10, batchSize=batchSize)


@pytest.mark.parametrize("count", [0, -1])
def test_nonpositive_sample_count_is_rejected(count):
    with pytest.raises(ValueError, match="nSamples"):
        HV(np.zeros((1, 4)), np.ones(4), nSamples=count)


def test_exact_low_dimensional_branch_does_not_sample():
    class NoSampling:
        def uniform(self, *args, **kwargs):
            pytest.fail("Exact HV must not sample")

    assert HV([[0.5, 0.5]], [1, 1], normalize=False, rng=NoSampling()) == 0.25
