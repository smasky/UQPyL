import numpy as np
from math import fsum
import warnings


def _prepare_obs_sim(obs, sim, mask=None):
    """
    Validate and align flattened observations, simulations, and masks.

    Args:
        obs: Observation vector with shape `(n_obs,)`.
        sim: Simulation array with shape `(n_samples, n_obs)` or `(n_obs,)`.
        mask: Optional boolean mask with shape `(n_obs,)`, where `True`
            indicates missing observations.

    Returns:
        tuple: `(obs_valid, sim_valid)` after mask filtering.
    """
    obs = np.asarray(obs, dtype=float).reshape(-1)
    sim = np.asarray(sim, dtype=float)

    if sim.ndim == 1:
        sim = sim.reshape(1, -1)
    if sim.ndim != 2:
        raise ValueError("sim must be a 1D or 2D array.")
    if sim.shape[1] != obs.size:
        raise ValueError("sim.shape[1] must equal obs.size.")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != obs.size:
            raise ValueError("mask size must equal obs.size.")
        valid = ~mask
        obs = obs[valid]
        sim = sim[:, valid]

    if obs.size == 0:
        raise ValueError("No valid observations remain after applying mask.")

    return obs, sim


def mse(obs, sim, mask=None):
    """Mean squared error for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    scaled, exponent = _scaledResidual(obs, sim)
    with np.errstate(under="ignore"):
        moment = np.mean(scaled**2, axis=1)
    return _restoreError(moment, 2 * exponent, "MSE")


def mae(obs, sim, mask=None):
    """Mean absolute error for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    scaled, exponent = _scaledResidual(obs, sim)
    return _restoreError(np.mean(np.abs(scaled), axis=1), exponent, "MAE")


def rmse(obs, sim, mask=None):
    """Compute RMSE without requiring the squared errors to be representable."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    scaled, exponent = _scaledResidual(obs, sim)
    with np.errstate(under="ignore"):
        rootMeanSquare = np.sqrt(np.mean(scaled**2, axis=1))
    return _restoreError(rootMeanSquare, exponent, "RMSE")


def _scaledResidual(obs, sim):
    """Represent residuals as bounded mantissas and per-row binary exponents."""
    with np.errstate(over="ignore"):
        residual = sim - obs.reshape(1, -1)
    # A finite difference can exceed double even when its RMS is representable.
    # Only those rows need half-scale subtraction; ordinary rows retain tiny
    # residuals alongside large, exactly matching observation values.
    overflowRows = np.all(np.isfinite(sim), axis=1) & np.all(np.isfinite(obs)) & np.any(np.isinf(residual), axis=1)
    with np.errstate(under="ignore"):
        residual[overflowRows] = sim[overflowRows] * 0.5 - obs * 0.5
        exponent = np.frexp(np.max(np.abs(residual), axis=1, keepdims=True))[1]
        scaled = np.ldexp(residual, -exponent)
    return scaled, exponent[:, 0] + overflowRows.astype(int)


def _restoreError(moment, exponent, name):
    """Restore physical units; only genuinely unrepresentable values warn."""
    with np.errstate(over="ignore", under="ignore"):
        result = np.ldexp(moment, exponent)
    outOfRange = np.isfinite(moment) & (moment > 0) & ((result == 0) | np.isinf(result))
    if np.any(outOfRange):
        warnings.warn(
            f"{name} exceeds floating-point range; underflow is returned as zero and overflow as infinity.",
            RuntimeWarning,
            stacklevel=3,
        )
    return result


def _scaleForMoments(values):
    """Scale each observation/simulation row by an exact binary power.

    The largest magnitude becomes at most one. Squared deviations then
    remain representable for ordinary finite variation, regardless of units.
    Unlike arbitrary scaling, binary powers preserve representable offsets.
    """
    exponent = np.frexp(np.max(np.abs(values), axis=-1, keepdims=True))[1]
    return np.ldexp(values, -exponent), exponent


def _sumForRatio(values):
    """Compensate cancellation before checking an observation sum or mean.

    The tolerance selects accurate summation; it never classifies a sum as
    zero. Only cancellation-prone finite rows need the scalar fallback.
    """
    rows = np.atleast_2d(values)
    totals = np.sum(rows, axis=1)
    errorBound = rows.shape[1] * np.finfo(float).eps * np.sum(np.abs(rows), axis=1)
    needsCompensation = (np.abs(totals) <= errorBound) & np.all(np.isfinite(rows), axis=1)
    for index in np.flatnonzero(needsCompensation):
        totals[index] = fsum(rows[index])
    return totals[0] if values.ndim == 1 else totals


def nse(obs, sim, mask=None):
    """Nash-Sutcliffe efficiency for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    if np.isfinite(obs[0]) and np.all(obs == obs[0]):
        raise ValueError("NSE is undefined when observation variance is zero.")
    obs, exponent = _scaleForMoments(obs)
    sim = np.ldexp(sim, -exponent)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0.0:
        raise ValueError("NSE is undefined when observation variance is zero.")
    sse = np.sum((sim - obs.reshape(1, -1)) ** 2, axis=1)
    return 1.0 - sse / denom


def r2(obs, sim, mask=None):
    """Coefficient of determination using the same formulation as NSE here."""
    return nse(obs, sim, mask=mask)


def pbias(obs, sim, mask=None):
    """Percent bias for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    obs, exponent = _scaleForMoments(obs)
    sim = np.ldexp(sim, -exponent)
    denom = _sumForRatio(obs)
    if denom == 0.0:
        raise ValueError("PBIAS is undefined when observation sum is zero.")
    return 100.0 * (_sumForRatio(sim - obs.reshape(1, -1)) / denom)


def pearson_r(obs, sim, mask=None):
    """Pearson correlation coefficient for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    if np.isfinite(obs[0]) and np.all(obs == obs[0]):
        raise ValueError("PearsonR is undefined when observation variance is zero.")
    if np.any(np.isfinite(sim[:, 0]) & np.all(sim == sim[:, :1], axis=1)):
        raise ValueError("PearsonR is undefined when a simulation variance is zero.")
    obs, _ = _scaleForMoments(obs)
    sim, _ = _scaleForMoments(sim)
    obs_centered = obs - np.mean(obs)
    obs_norm = np.linalg.norm(obs_centered)
    if obs_norm == 0.0:
        raise ValueError("PearsonR is undefined when observation variance is zero.")

    sim_centered = sim - np.mean(sim, axis=1, keepdims=True)
    sim_norm = np.linalg.norm(sim_centered, axis=1)
    if np.any(sim_norm == 0.0):
        raise ValueError("PearsonR is undefined when a simulation variance is zero.")

    return np.sum(sim_centered * obs_centered.reshape(1, -1), axis=1) / (sim_norm * obs_norm)


def kge(obs, sim, mask=None):
    """Kling-Gupta efficiency for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    r = pearson_r(obs, sim)
    obs, obsExponent = _scaleForMoments(obs)
    sim, simExponent = _scaleForMoments(sim)
    obs_mean = _sumForRatio(obs) / obs.size
    sim_mean = _sumForRatio(sim) / obs.size
    obs_std = np.std(obs)
    sim_std = np.std(sim, axis=1)

    if obs_mean == 0.0:
        raise ValueError("KGE is undefined when observation mean is zero.")
    if obs_std == 0.0:
        raise ValueError("KGE is undefined when observation standard deviation is zero.")

    exponentDifference = (simExponent - obsExponent).ravel()
    beta = np.ldexp(sim_mean / obs_mean, exponentDifference)
    alpha = np.ldexp(sim_std / obs_std, exponentDifference)
    return 1.0 - np.hypot(np.hypot(r - 1.0, alpha - 1.0), beta - 1.0)


def _prepareInterval(obs, lower, upper, mask=None):
    """Align intervals and reject empty or reversed unmasked bounds."""
    obs = np.asarray(obs, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    if not (lower.size == upper.size == obs.size):
        raise ValueError("obs, lower, and upper must have the same size.")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != obs.size:
            raise ValueError("mask size must equal obs.size.")
        valid = ~mask
        obs = obs[valid]
        lower = lower[valid]
        upper = upper[valid]

    if obs.size == 0:
        raise ValueError("No valid observations remain after applying mask.")
    if np.any(lower > upper):
        raise ValueError("Interval lower bounds must not exceed upper bounds.")
    return obs, lower, upper


def pfactor(obs, lower, upper, mask=None):
    """Coverage ratio, including endpoints; reject empty or reversed intervals."""
    obs, lower, upper = _prepareInterval(obs, lower, upper, mask)
    return float(np.mean((obs >= lower) & (obs <= upper)))


def rfactor(obs, lower, upper, mask=None):
    """Mean width / observation std; reject empty, reversed or zero-std inputs."""
    obs, lower, upper = _prepareInterval(obs, lower, upper, mask)
    if np.isfinite(obs[0]) and np.all(obs == obs[0]):
        raise ValueError("RFactor is undefined when observation standard deviation is zero.")
    obs, exponent = _scaleForMoments(obs)
    lower, upper = np.ldexp(lower, -exponent), np.ldexp(upper, -exponent)
    obs_std = np.std(obs)
    if obs_std == 0.0:
        raise ValueError("RFactor is undefined when observation standard deviation is zero.")
    return float(np.mean(upper - lower) / obs_std)
