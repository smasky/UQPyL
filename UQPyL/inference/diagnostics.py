"""Explicit post-run diagnostics; no work is added to the sampling loop.

Rank/folded R-hat and bulk/tail ESS follow Vehtari et al. (2021),
https://doi.org/10.1214/20-BA1221. ESS uses FFT autocovariances and Geyer's
initial positive/monotone sequence, including the antithetic-chain correction.
"""

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import ndtri
from scipy.stats import rankdata


def _splitChains(values):
    half = values.shape[1] // 2
    return np.concatenate((values[:, :half], values[:, -half:]), axis=0)


def _rankNormalize(values):
    ranks = rankdata(values, method="average").reshape(values.shape)
    return ndtri((ranks - 0.375) / (values.size + 0.25))


def _rhat(values):
    within = np.mean(np.var(values, axis=1, ddof=1))
    if within == 0:
        return None
    betweenMeans = np.var(np.mean(values, axis=1), ddof=1)
    return float(np.hypot(np.sqrt(1 - 1 / values.shape[1]), np.sqrt(betweenMeans) / np.sqrt(within)))


def _effectiveSize(values):
    """Multi-chain ESS for already split, finite, nonconstant draws."""
    nDraws = values.shape[1]
    centered = values - values.mean(axis=1, keepdims=True)
    fftSize = next_fast_len(2 * nDraws)
    spectrum = rfft(centered, n=fftSize, axis=1)
    autocov = irfft(spectrum * spectrum.conjugate(), n=fftSize, axis=1)[:, :nDraws] / nDraws
    within = autocov[:, 0].mean() * nDraws / (nDraws - 1)
    marginal = within * (1 - 1 / nDraws) + np.var(values.mean(axis=1), ddof=1)
    if marginal <= 0:
        return None
    rho = 1 - (within - autocov.mean(axis=0)) / marginal
    rho[0] = 1
    accepted = np.zeros(nDraws)
    accepted[:2] = rho[:2]
    pairStart, lastEven = 2, 1.0
    # Retain the initial nonnegative adjacent-lag pairs only. Leave one lag
    # beyond the final complete pair for the finite-sample correction.
    while pairStart + 1 < nDraws - 1 and accepted[pairStart - 2 : pairStart].sum() > 0:
        lastEven = rho[pairStart]
        if rho[pairStart : pairStart + 2].sum() >= 0:
            accepted[pairStart : pairStart + 2] = rho[pairStart : pairStart + 2]
        pairStart += 2
    lastPaired = pairStart - 3
    if lastEven > 0:
        accepted[lastPaired + 1] = lastEven
    pairs = accepted[: lastPaired + 1].reshape(-1, 2).sum(axis=1)
    monotonePairs = np.minimum.accumulate(pairs)
    tau = -1 + 2 * monotonePairs.sum() + accepted[lastPaired + 1]
    # Negative correlation may give ESS > draw count; do not cap at N.
    tau = max(float(tau), 1 / np.log10(values.size))
    return float(values.size / tau)


def _appendMetric(metric, value=None, status="available"):
    if value is not None and not np.isfinite(value):
        value, status = None, "numerical_failure"
    metric["values"].append(value)
    metric["status"].append(status)


def computeChainDiagnostics(decs):
    """Compute classical/modern R-hat and bulk/tail ESS on formal draws.

    Args:
        decs: Real-coordinate array (n_chains, n_draws, n_input), excluding
            warm-up. At least four draws are needed; R-hat needs two chains,
            ESS can use one. Odd-length splits omit the middle draw.

    Returns:
        dict: Independent per-variable values and availability statuses.
            rhat is the maximum of rank-normalized and folded split R-hat.
            ess_tail is the minimum ESS of 5%/95% quantile indicators; cutoffs
            use all formal draws before splitting. Constant chains or tails
            yield None, not misleading full-sample ESS. No convergence verdict
            is inferred and no sampling state, RNG, or stopping rule changes.
    """
    samples = np.asarray(decs)
    if samples.ndim != 3:
        raise ValueError("decs must have shape (n_chains, n_draws, n_input).")
    nChains, nDraws, nInput = samples.shape
    metrics = {
        "split_rhat": {"method": "classical_split", "draws_per_half": nDraws // 2},
        "rhat": {"method": "rank_normalized_folded_split"},
        "ess_bulk": {"method": "rank_normalized_split"},
        "ess_tail": {"method": "quantile_split", "probabilities": [0.05, 0.95]},
    }
    for metric in metrics.values():
        metric.update(values=[], status=[])
    for index in range(nInput):
        reason = None
        column = samples[:, :, index]
        if nChains == 0:
            reason = "insufficient_chains"
        elif nDraws < 4:
            reason = "insufficient_draws"
        elif np.iscomplexobj(column):
            reason = "non_numeric"
        else:
            try:
                column = column.astype(float)
            except (TypeError, ValueError, OverflowError):
                reason = "non_numeric"
            else:
                if not np.all(np.isfinite(column)):
                    reason = "nonfinite"
                elif np.any(np.all(column == column[:, :1], axis=1)):
                    reason = "constant_chain"
        if reason is not None:
            for name, metric in metrics.items():
                status = "insufficient_chains" if nChains < 2 and "rhat" in name else reason
                _appendMetric(metric, status=status)
            continue

        # Power-of-two scaling protects folding/quantiles/variance at extreme
        # magnitudes. Rank normalization uses the original values to keep ties.
        exponent = np.frexp(np.max(np.abs(column)))[1]
        scaled = np.ldexp(column, -int(exponent))
        halves = _splitChains(scaled)
        ranked = _rankNormalize(_splitChains(column))
        if nChains < 2:
            for name in ("split_rhat", "rhat"):
                _appendMetric(metrics[name], status="insufficient_chains")
        else:
            classic = _rhat(halves) if np.all(np.var(halves, axis=1) > 0) else None
            _appendMetric(metrics["split_rhat"], classic, "available" if classic is not None else "constant_chain")
            bulkRhat = _rhat(ranked)
            folded = _rankNormalize(np.abs(halves - np.median(halves)))
            foldedRhat = _rhat(folded)
            if bulkRhat is None or foldedRhat is None:
                _appendMetric(metrics["rhat"], status="constant_split" if bulkRhat is None else "constant_folded")
            else:
                _appendMetric(metrics["rhat"], max(bulkRhat, foldedRhat))

        bulkEss = _effectiveSize(ranked)
        _appendMetric(metrics["ess_bulk"], bulkEss, "available" if bulkEss is not None else "constant_split")
        tails = []
        for quantile in np.quantile(scaled, [0.05, 0.95]):
            indicator = _splitChains((scaled <= quantile).astype(float))
            if np.all(indicator == indicator.flat[0]):
                tails.append(None)
            else:
                tails.append(_effectiveSize(indicator))
        if any(value is None for value in tails):
            _appendMetric(metrics["ess_tail"], status="constant_tail")
        else:
            _appendMetric(metrics["ess_tail"], min(tails))
    return {"sample_scope": "formal_draws", "n_chains": nChains, "n_draws": nDraws, **metrics}
