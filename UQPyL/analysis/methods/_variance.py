import numpy as np


def scaleOutput(values):
    """Center and scale one output column/block without squaring raw magnitudes."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Sensitivity analysis requires finite output values.")
    magnitude = np.max(np.abs(values))
    if magnitude == 0:
        return np.zeros_like(values)

    # Power-of-two scaling protects subtraction and preserves close float values.
    exponent = int(np.frexp(magnitude)[1])
    scaled = np.ldexp(values, -exponent)
    shifted = scaled - scaled.flat[0]
    spread = np.max(np.abs(shifted))
    if spread == 0:
        return np.zeros_like(values)

    shifted /= spread
    return shifted - np.mean(shifted)
