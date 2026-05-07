import numpy as np


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
    return np.mean((sim - obs.reshape(1, -1)) ** 2, axis=1)


def mae(obs, sim, mask=None):
    """Mean absolute error for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    return np.mean(np.abs(sim - obs.reshape(1, -1)), axis=1)


def rmse(obs, sim, mask=None):
    """Root mean squared error for one or more simulation rows."""
    return np.sqrt(mse(obs, sim, mask=mask))


def nse(obs, sim, mask=None):
    """Nash-Sutcliffe efficiency for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if np.isclose(denom, 0.0):
        raise ValueError("NSE is undefined when observation variance is zero.")
    sse = np.sum((sim - obs.reshape(1, -1)) ** 2, axis=1)
    return 1.0 - sse / denom


def r2(obs, sim, mask=None):
    """Coefficient of determination using the same formulation as NSE here."""
    return nse(obs, sim, mask=mask)


def pbias(obs, sim, mask=None):
    """Percent bias for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    denom = np.sum(obs)
    if np.isclose(denom, 0.0):
        raise ValueError("PBIAS is undefined when observation sum is zero.")
    return 100.0 * np.sum(sim - obs.reshape(1, -1), axis=1) / denom


def pearson_r(obs, sim, mask=None):
    """Pearson correlation coefficient for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    obs_centered = obs - np.mean(obs)
    obs_norm = np.linalg.norm(obs_centered)
    if np.isclose(obs_norm, 0.0):
        raise ValueError("PearsonR is undefined when observation variance is zero.")

    sim_centered = sim - np.mean(sim, axis=1, keepdims=True)
    sim_norm = np.linalg.norm(sim_centered, axis=1)
    if np.any(np.isclose(sim_norm, 0.0)):
        raise ValueError("PearsonR is undefined when a simulation variance is zero.")

    return np.sum(sim_centered * obs_centered.reshape(1, -1), axis=1) / (sim_norm * obs_norm)


def kge(obs, sim, mask=None):
    """Kling-Gupta efficiency for one or more simulation rows."""
    obs, sim = _prepare_obs_sim(obs, sim, mask=mask)
    r = pearson_r(obs, sim)
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim, axis=1)
    obs_std = np.std(obs)
    sim_std = np.std(sim, axis=1)

    if np.isclose(obs_mean, 0.0):
        raise ValueError("KGE is undefined when observation mean is zero.")
    if np.isclose(obs_std, 0.0):
        raise ValueError("KGE is undefined when observation standard deviation is zero.")

    beta = sim_mean / obs_mean
    alpha = sim_std / obs_std
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def pfactor(obs, lower, upper, mask=None):
    """Coverage ratio of observations inside the predictive interval."""
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

    return float(np.mean((obs >= lower) & (obs <= upper)))


def rfactor(obs, lower, upper, mask=None):
    """Average predictive interval width normalized by observation standard deviation."""
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

    obs_std = np.std(obs)
    if np.isclose(obs_std, 0.0):
        raise ValueError("RFactor is undefined when observation standard deviation is zero.")
    return float(np.mean(upper - lower) / obs_std)
