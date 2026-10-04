"""Binary scaling for objective geometry, without changing objective units relatively."""

import numpy as np
import warnings


def shiftedObjectives(values, *, perColumn=False):
    """Return (values - column minima) / 2**exponent without overflow.

    A common exponent preserves angles and relative objective weights. A vector
    of exponents is available for independently normalized crowding distances.
    Subtract first to retain tiny variation beside large constant objectives.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Objective geometry requires a finite matrix.")
    minimum = np.min(values, axis=0)
    with np.errstate(over="ignore", under="ignore"):
        delta = values - minimum
        half = np.any(np.isinf(delta), axis=0)
        delta[:, half] = values[:, half] * 0.5 - minimum[half] * 0.5
        span = np.max(delta, axis=0)
        exponents = np.frexp(span)[1] + half.astype(int)
        exponent = exponents if perColumn else int(np.max(exponents[span > 0], initial=-1074))
        scaled = np.ldexp(delta, half.astype(int) - exponent)
    return scaled, exponent


def unitVectors(values):
    """Normalize finite nonzero rows without squaring physical magnitudes."""
    values = np.asarray(values, dtype=float)
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(scale == 0):
        raise ValueError("Reference vectors must be finite and nonzero.")
    scaled = values / scale
    return scaled / np.hypot.reduce(scaled, axis=1, keepdims=True)


def automaticReference(worst, margin):
    """Construct a finite HV reference and report an unrepresentable margin."""
    worst = np.asarray(worst, dtype=float)
    with np.errstate(over="ignore"):
        reference = worst + margin * np.where(worst == 0, 1.0, np.abs(worst))
    if np.any(np.isposinf(reference)):
        warnings.warn(
            "Automatic hypervolume reference exceeds floating-point range; "
            "clipping its margin. Supply an explicit reference point for interpretable HV.",
            RuntimeWarning,
            stacklevel=3,
        )
        reference = np.minimum(reference, np.finfo(float).max)
    return reference
