"""Binary scaling for surrogate moments and dimensionless scores."""

import numpy as np


def centeredColumns(values):
    """Return centered columns, their binary powers, and scaled means."""
    powers = np.frexp(np.max(np.abs(values), axis=0))[1]
    scaled = np.ldexp(values, -powers)
    # Subtract an anchor before averaging to retain small representable spread
    # around a large offset and make constant columns exactly zero.
    shifted = scaled - scaled[0]
    meanShift = np.mean(shifted, axis=0)
    return shifted - meanShift, powers, scaled[0] + meanShift


def squaredSum(values, powers):
    """Represent a sum of squared columns as mantissa * 2**power."""
    fractions, localPowers = np.frexp(np.sum(values**2, axis=0))
    active = fractions != 0
    if not np.any(active):
        return 0.0, 0
    totalPowers = localPowers[active] + 2 * powers[active]
    power = int(np.max(totalPowers))
    return float(np.sum(np.ldexp(fractions[active], totalPowers - power))), power
