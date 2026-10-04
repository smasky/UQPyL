"""Explicit likelihood weighting and inverse empirical-CDF summaries."""

import numpy as np


def likelihoodWeights(obs, simulations, mask, logLikelihood):
    count = len(simulations)
    if logLikelihood is None:
        return np.full(count, 1.0 / count)
    logWeights = np.asarray(logLikelihood(obs.copy(), simulations.copy(), mask=mask.copy()), dtype=float)
    if logWeights.shape != (count,) or np.any(np.isnan(logWeights)) or np.any(np.isposinf(logWeights)):
        raise ValueError("logLikelihood must return one finite or -inf value per sample.")
    if not np.any(np.isfinite(logWeights)):
        raise ValueError("At least one sample must have positive likelihood.")
    weights = np.exp(logWeights - np.max(logWeights))
    return weights / weights.sum()


def weightedQuantiles(values, weights, interval):
    probabilities = [(1 - interval) / 2, (1 + interval) / 2]
    bounds = np.empty((2, values.shape[1]))
    positive = weights > 0
    for column in range(values.shape[1]):
        selected, mass = values[positive, column], weights[positive]
        order = np.argsort(selected, kind="stable")
        cumulative = np.cumsum(mass[order])
        cumulative[-1] = 1.0
        bounds[:, column] = selected[order[np.searchsorted(cumulative, probabilities, side="left")]]
    return bounds
