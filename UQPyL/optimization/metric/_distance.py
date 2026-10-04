"""Bounded-memory nearest-distance means with full floating-point range."""

import warnings
import numpy as np


def meanNearestDistance(points, reference):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    if not points.size or not reference.size:
        raise ValueError("Both point sets must contain at least one point.")
    if points.ndim != 2 or reference.ndim != 2 or points.shape[1] != reference.shape[1]:
        raise ValueError("Point sets must be matrices with matching objective dimensions.")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(reference)):
        raise ValueError("Point sets must contain finite values.")
    bestMantissa = np.ones(len(points))
    bestExponent = np.full(len(points), np.iinfo(np.int32).max)
    for start in range(0, len(points), 128):
        query = points[start : start + 128]
        mantissa = bestMantissa[start : start + len(query)]
        exponent = bestExponent[start : start + len(query)]
        for refStart in range(0, len(reference), 128):
            target = reference[refStart : refStart + 128]
            with np.errstate(over="ignore", under="ignore"):
                delta = query[:, None, :] - target[None, :, :]
                half = np.any(np.isinf(delta), axis=2)
                if np.any(half):
                    halfDelta = query[:, None, :] * 0.5 - target[None, :, :] * 0.5
                    delta[half] = halfDelta[half]
                powers = np.frexp(np.max(np.abs(delta), axis=2))[1]
                scaled = np.ldexp(delta, -powers[:, :, None])
                norms = np.hypot.reduce(scaled, axis=2)
                fractions, normPowers = np.frexp(norms)
                powers = powers + normPowers + half.astype(int)
                powers[fractions == 0] = -1075
            lowest = np.min(powers, axis=1)
            nearest = np.min(np.where(powers == lowest[:, None], fractions, np.inf), axis=1)
            better = (lowest < exponent) | ((lowest == exponent) & (nearest < mantissa))
            exponent[better] = lowest[better]
            mantissa[better] = nearest[better]
    common = int(np.max(bestExponent))
    with np.errstate(over="ignore", under="ignore"):
        mean = np.mean(np.ldexp(bestMantissa, bestExponent - common))
        result = np.ldexp(mean, common)
    if mean > 0 and (result == 0 or np.isinf(result)):
        warnings.warn(
            "Mean distance exceeds floating-point range; returning zero or infinity.", RuntimeWarning, stacklevel=3
        )
    return float(result)
