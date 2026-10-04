import numpy as np
import warnings


def scaleOutput(values, *, fitRows=None, returnScale=False):
    """Center/scale an output block, optionally fitting only selected rows.

    returnScale also returns (spread, exponent), giving the physical output
    scale as spread * 2**exponent without requiring that product to be finite.
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Sensitivity analysis requires finite output values.")
    fitted = values if fitRows is None else values[fitRows]
    magnitude = np.max(np.abs(fitted))
    if magnitude == 0:
        zeros = np.zeros_like(values)
        return (zeros, (0.0, 0)) if returnScale else zeros

    # Power-of-two scaling protects subtraction and preserves close float values.
    exponent = int(np.frexp(magnitude)[1])
    scaled = np.ldexp(values, -exponent)
    if fitRows is None:
        shifted = scaled - scaled.flat[0]
    else:
        # Retain MARS's mean-centered/max-deviation amplitude. Power-of-two
        # preprocessing protects the mean while preserving ordinary fit paths.
        shifted = scaled - np.mean(scaled[fitRows])
    fittedShift = shifted if fitRows is None else shifted[fitRows]
    spread = np.max(np.abs(fittedShift))
    if spread == 0:
        zeros = np.zeros_like(values)
        return (zeros, (0.0, exponent)) if returnScale else zeros

    shifted /= spread
    result = shifted - np.mean(shifted) if fitRows is None else shifted
    return (result, (float(spread), exponent)) if returnScale else result


def restoreSquaredOutput(values, scale, methodName, *, warnUnderflow=True):
    """Restore squared units without forming an overflowing/underflowing scale².

    The physical output scale is spread * 2**exponent. Separate mantissas
    and powers also retain a tiny scaled score whose physical value is finite.
    """
    values = np.asarray(values, dtype=float)
    spread, exponent = scale
    spread, spreadPower = np.frexp(spread)
    exponent += int(spreadPower)
    fractions, powers = np.frexp(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        restored = np.ldexp(fractions * spread * spread, powers + 2 * exponent)
    if not np.all(np.isfinite(restored)):
        raise ValueError(f"{methodName} scores exceed the finite squared-output range.")
    underflow = bool(np.any((values != 0) & (restored == 0)))
    if underflow and warnUnderflow:
        warnings.warn(
            f"{methodName} raw scores underflow in squared output units; "
            "normalized scores were computed before restoring physical units and remain available.",
            RuntimeWarning,
            stacklevel=3,
        )
    return restored, underflow


def scaleOutputColumns(values):
    """Center columns independently, using one common scale to retain weights."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Sensitivity analysis requires finite output values.")
    # Center each column in its own power-of-two units first. A huge constant
    # output must not erase a tiny varying column before the common scaling.
    exponents = np.frexp(np.max(np.abs(values), axis=0))[1]
    shifted = np.ldexp(values, -exponents)
    shifted -= shifted[:1]
    columnSpreads = np.max(np.abs(shifted), axis=0)
    varying = columnSpreads > 0
    if not np.any(varying):
        return np.zeros_like(values), (0.0, 0)
    powers = np.frexp(columnSpreads)[1] + exponents
    exponent = int(np.max(powers[varying]))
    shifted = np.ldexp(shifted, exponents - exponent)
    spread = float(np.max(np.abs(shifted)))
    shifted /= spread
    return shifted - np.mean(shifted, axis=0), (spread, exponent)
