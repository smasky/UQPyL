"""MARS must not read inactive rows of an uninitialized QR workspace."""

import numpy as np
from UQPyL.surrogate.mars.core._knot_search import SingleWeightDependentData, SingleOutcomeDependentData


def testMarsInitialOutcomeIgnoresInactiveQrRows():
    weights = SingleWeightDependentData.alloc(np.ones(4), 4, 3, 1e-12)
    # Poison unused storage deterministically rather than relying on allocator
    # history; inf*0 exposes any attempted projection before a basis exists.
    np.asarray(weights.Q_t)[:] = np.inf
    y = np.array([0.0, 1.0, 2.0, 4.0])
    with np.errstate(invalid="raise"):
        outcome = SingleOutcomeDependentData.alloc(y, weights, 4, 3)
    np.testing.assert_array_equal(outcome.theta, np.zeros(3))
    assert outcome.sse_ == 21
    weights.update_from_array(np.ones(4))
    outcome.update()
    assert np.isclose(outcome.sse_, 8.75)
